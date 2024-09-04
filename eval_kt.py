"""
Usage:
python eval_kt.py \
    -k data/outputs/2024.03.02/21.10.08_train_keypose_transformer_keypose_sim_transfer_cube_scripted/checkpoints/latest.ckpt \
    -t data/outputs/2024.03.02/20.40.33_train_trajectory_transformer_trajectory_sim_transfer_cube_scripted/checkpoints/latest.ckpt \
    -o data/eval/sim_transfer_cube_scripted/kt/ \
    -c diffusion_policy/config/task/sim_transfer_cube_scripted.yaml \
    -e 0.35
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import copy
import pathlib
import click
import hydra
import torch
import dill
import wandb
import json
import time
from omegaconf import OmegaConf

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.kt_policy import KeyposeTrajectoryPolicy

@click.command()
@click.option('-k', '--keypose_checkpoint', required=True)
@click.option('-t', '--trajectory_checkpoint', required=True)
@click.option('-c', '--task_cfg', type=click.Path(exists=True))
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
@click.option('-e', '--epsilon', type=float, default=0.25)
def main(
    keypose_checkpoint,
    trajectory_checkpoint,
    output_dir,
    device,
    task_cfg,
    epsilon
):
    output_dir += time.strftime("%Y.%m.%d/%H.%M.%S", time.localtime())
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    ## some say that these settings increase accuracy
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    ### load keypose checkpoint
    k_payload = torch.load(open(keypose_checkpoint, 'rb'), pickle_module=dill)
    k_cfg = k_payload['cfg']

    cls = hydra.utils.get_class(k_cfg._target_)
    k_workspace = cls(k_cfg, output_dir=output_dir)
    k_workspace: BaseWorkspace
    ### in case that model & opt are not defined in __init__ (e.g. ddp)
    if "model" not in k_workspace.__dict__.keys():
        k_workspace.model = hydra.utils.instantiate(k_cfg.policy)
    if "optimizer" not in k_workspace.__dict__.keys():
        k_workspace.optimizer = k_workspace.model.get_optimizer(**k_cfg.optimizer)
    k_workspace.load_payload(k_payload, exclude_keys=None, include_keys=None)

    k_policy = k_workspace.model
    device = torch.device(device)
    k_policy.to(device)
    k_policy.eval()
    print("keypose policy loaded")

    ### load trajectory checkpoint
    t_payload = torch.load(open(trajectory_checkpoint, 'rb'), pickle_module=dill)
    t_cfg = t_payload['cfg']

    cls = hydra.utils.get_class(t_cfg._target_)
    t_workspace = cls(t_cfg, output_dir=output_dir)
    t_workspace: BaseWorkspace
    ### in case that model, ema_model & opt are not defined in __init__ (e.g. ddp)
    if "model" not in t_workspace.__dict__.keys():
        t_workspace.model = hydra.utils.instantiate(t_cfg.policy)
    if "ema_model" not in t_workspace.__dict__.keys() and t_cfg.training.use_ema:
        t_workspace.ema_model = copy.deepcopy(t_workspace.model)
    if "optimizer" not in t_workspace.__dict__.keys():
        t_workspace.optimizer = t_workspace.model.get_optimizer(**t_cfg.optimizer)
    t_workspace.load_payload(t_payload, exclude_keys=None, include_keys=None)

    t_policy = t_workspace.model
    if t_cfg.training.use_ema:
        t_policy = t_workspace.ema_model
    
    t_policy.to(device)
    t_policy.eval()
    print("trajectory policy loaded")

    ### kt policy
    policy = KeyposeTrajectoryPolicy(k_policy, t_policy, epsilon=epsilon)

    ### init env_runner and run eval
    with open(task_cfg, 'r') as f:
        cfg = OmegaConf.load(f)
    # some modification on runner cfg
    cfg.env_runner.n_obs_steps = t_cfg.n_obs_steps
    cfg.env_runner.n_action_steps = t_cfg.n_action_steps
    cfg.env_runner.past_action = False
    cfg.env_runner.n_test = 50
    cfg.env_runner.n_test_vis = 0

    env_runner = hydra.utils.instantiate(
        cfg.env_runner,
        output_dir=output_dir)
    runner_log = env_runner.run(policy) ## 重点
    
    # dump log to json
    json_log = dict()
    json_log["checkpoint_keypose"] = keypose_checkpoint
    json_log["checkpoint_trajectory"] = trajectory_checkpoint
    json_log["eposilon"] = epsilon
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            json_log[key] = value
    out_path = os.path.join(output_dir, 'eval_log.json')
    json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)

if __name__ == '__main__':
    main()
