if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
import math

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.trajectory_transformer_policy import TrajectoryTransformerPolicy
from diffusion_policy.dataset.trajectory_base_dataset import TrajectoryBaseDataset
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, find_batch_size
from diffusion_policy.common.ddp_util import (
    init_distributed_mode, NoOpContextManager, reduce_across_processes
)
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainTrajectoryTransformerWorkspaceDDP(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        torch.backends.cudnn.benckmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True

        # configure training state
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # init distributed mode
        ddp_info = init_distributed_mode()
        rank = dist.get_rank()
        print(ddp_info)

        # configure dataset
        dataset: TrajectoryBaseDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, TrajectoryBaseDataset)
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        train_dataloader = DataLoader(
            dataset, sampler=train_sampler, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
        val_dataloader = DataLoader(
            val_dataset, sampler=val_sampler, **cfg.val_dataloader)

        # configure model and optimizer under DDP
        model: TrajectoryTransformerPolicy = hydra.utils.instantiate(cfg.policy)
        if cfg.training.freeze_encoder:
            model.obs_encoder.eval()
            model.obs_encoder.requires_grad_(False)
        model.set_normalizer(normalizer)

        ema_model: TrajectoryTransformerPolicy = None
        if cfg.training.use_ema:
            ema_model = copy.deepcopy(model)
            ema_model.set_normalizer(normalizer)

        device = torch.device(cfg.training.device)
        model.to(device)
        if ema_model is not None:
            ema_model.to(device)

        model = DDP(
            model,
            device_ids=[ddp_info["gpu"]],
            find_unused_parameters=True
        )
        ema_model = DDP(
            ema_model,
            device_ids=[ddp_info["gpu"]],
            find_unused_parameters=True
        ) if ema_model is not None else None
        self.model = model.module  # model without DDP wrapper
        self.ema_model = ema_model.module if ema_model is not None else None
        self.optimizer = self.model.get_optimizer(**cfg.optimizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(len(train_dataloader) * cfg.training.num_epochs),
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

        # configure logging for master only
        if rank == 0:
            assert rank == ddp_info["rank"] == 0
            wandb_run = wandb.init(
                dir=str(self.output_dir),
                config=OmegaConf.to_container(cfg, resolve=True),
                **cfg.logging
            )
            wandb.config.update(
                {
                    "output_dir": self.output_dir,
                }
            )

            # configure checkpoint
            topk_manager = TopKCheckpointManager(
                save_dir=os.path.join(self.output_dir, 'checkpoints'),
                **cfg.checkpoint.topk
            )

        # save batch for sampling
        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 10
            cfg.training.max_val_steps = 10
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1


        # training loop
        # init tqdm and logger for master only
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        json_logger = JsonLogger(log_path) if rank == 0 \
            else NoOpContextManager()      

        with json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
                # ========= train for this epoch ==========
                train_losses = list()
                tepoch = NoOpContextManager(iterable=train_dataloader)
                if rank == 0:
                    tepoch = tqdm.tqdm(
                        train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec
                    )
                with tepoch:
                    # set epoch for sampler
                    train_sampler.set_epoch(local_epoch_idx)

                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        # compute loss
                        self.optimizer.zero_grad()
                        loss = self.model.compute_loss(batch)
                        loss.backward()

                        # step optimizer
                        self.optimizer.step()
                        lr_scheduler.step()

                        # update ema
                        if cfg.training.use_ema:
                            ema.step(self.model)

                        # reduce loss across processes
                        local_bsz = find_batch_size(batch)
                        local_loss_sum = loss.item() * local_bsz
                        global_bsz = reduce_across_processes(local_bsz)
                        global_loss_sum = reduce_across_processes(local_loss_sum)
                        assert global_bsz > 0
                        loss_cpu = (global_loss_sum / global_bsz).item()

                        if rank == 0:
                            tepoch.set_postfix(loss=loss_cpu, refresh=False)
                        train_losses.append(loss_cpu)
                        step_log = {
                            'train_loss': loss_cpu,
                            'train_log_loss': math.log(loss_cpu),
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            if rank == 0:
                                # log of last step is combined with validation and rollout
                                wandb_run.log(step_log, step=self.global_step)
                                json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break

                # at the end of each epoch
                # replace train_loss with epoch average
                # train_loss = np.mean(train_losses)
                # step_log['train_loss'] = train_loss

                # ========= eval for this epoch ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run validation, reduce across processes
                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.inference_mode():
                        val_losses = list()
                        tepoch = NoOpContextManager(iterable=val_dataloader)
                        if rank == 0:
                            tepoch = tqdm.tqdm(
                                val_dataloader, desc=f"Validation epoch {self.epoch}", 
                                leave=False, mininterval=cfg.training.tqdm_interval_sec
                            )
                        with tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                loss = self.model.compute_loss(batch)
                                
                                # reduce loss across processes
                                local_bsz = find_batch_size(batch)
                                local_loss_sum = loss.item() * local_bsz
                                global_bsz = reduce_across_processes(local_bsz)
                                global_loss_sum = reduce_across_processes(local_loss_sum)
                                assert global_bsz > 0
                                loss_cpu = (global_loss_sum / global_bsz).item()

                                val_losses.append(loss_cpu)
                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            # log epoch average validation loss
                            step_log['val_loss'] = val_loss
                            step_log['val_log_loss'] = math.log(val_loss)

                # run diffusion sampling on a training batch, then reduce
                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.inference_mode():
                        # sample trajectory from training set, and evaluate difference
                        batch = dict_apply(
                            train_sampling_batch, 
                            lambda x: x.to(device, non_blocking=True)
                        )
                        obs_dict = batch['obs']
                        next_keypose = batch['next_keypose']
                        gt_action = batch['action']
                        
                        result = policy.predict_trajectory(obs_dict, next_keypose)
                        pred_action = result['action_pred']
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)

                        # reduce loss across processes
                        local_bsz = find_batch_size(batch)
                        local_mse_sum = mse.item() * local_bsz
                        global_bsz = reduce_across_processes(local_bsz)
                        global_mse_sum = reduce_across_processes(local_mse_sum)
                        assert global_bsz > 0
                        mse_cpu = (global_mse_sum / global_bsz).item()

                        step_log['train_action_mse_error'] = mse_cpu
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse
                
                # checkpoint, master only
                if (self.epoch % cfg.training.checkpoint_every) == 0 and rank == 0:
                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value
                    
                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)
                # ========= eval end for this epoch ==========
                self.model.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                # master only
                if rank == 0:
                    wandb_run.log(step_log, step=self.global_step)
                    json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

                dist.barrier()

        # end of training
        dist.destroy_process_group()

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainTrajectoryTransformerWorkspaceDDP(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
