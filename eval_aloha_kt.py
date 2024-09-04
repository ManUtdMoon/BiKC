"""
Usage:
python eval_aloha_kt.py \
    -k data/outputs/2024.04.19/14.53.26_train_keypose_transformer_keypose_aloha_insert_10s_random_init/checkpoints/latest.ckpt \
    -t data/outputs/2024.04.20/11.41.59_train_trajectory_transformer_trajectory_aloha_insert_10s_random_init/checkpoints/latest.ckpt \
    -o data/eval/aloha_insert_10s_random_init/ \
    -md 500 \
    -n 20 \
    -s 4 \
    -d cuda:0 \
    -e 0.5
"""

import os
import time
import numpy as np
import click
import copy
import numpy as np
import torch
import dill
import hydra
import pathlib
import json
from omegaconf import OmegaConf
from einops import rearrange

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.kt_policy import KeyposeTrajectoryPolicy
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecorder

from aloha.aloha_scripts.robot_utils import move_grippers
from aloha.aloha_scripts.real_env import make_real_env


OmegaConf.register_new_resolver("eval", eval, replace=True)

## data configuration settings
@click.command()
@click.option('--keypose_ckpt', '-k', required=True, help='Path to keypose checkpoint')
@click.option('--trajectory_ckpt', '-t', required=True, help='Path to traj checkpoint')
@click.option('--output', '-o', required=True, help='Directory to save recording')
@click.option('--max_timesteps', '-md', default=500, help='Max duration for each epoch in seconds.')
@click.option('--num_inference_steps', '-n', default=16, type=int, help="DDIM inference iterations.")
@click.option('--scale', '-s', default=4, type=int, help="Image downsample scale")
@click.option('--device', '-d', default="cuda:0")
@click.option('--epsilon', '-e', type=float, default=0.5)
def main(
    keypose_ckpt, 
    trajectory_ckpt,
    output,
    max_timesteps,
    num_inference_steps,
    scale,
    device,
    epsilon
):
    output += time.strftime("%Y.%m.%d/%H.%M.%S", time.localtime())
    if os.path.exists(output):
        click.confirm(f"Output path {output} already exists! Overwrite?", abort=True)
    pathlib.Path(output).mkdir(parents=True, exist_ok=True)

    ##############################
    #       load checkpoints     #
    ##############################
    # load keypose checkpoint
    k_payload = torch.load(open(keypose_ckpt, 'rb'), pickle_module=dill)
    k_cfg = k_payload['cfg']

    cls = hydra.utils.get_class(k_cfg._target_)
    k_workspace = cls(k_cfg, output_dir=output)
    k_workspace: BaseWorkspace
    ## in case that model & opt are not defined in __init__ (e.g. ddp)
    if "model" not in k_workspace.__dict__.keys():
        k_workspace.model = hydra.utils.instantiate(k_cfg.policy)
    if "optimizer" not in k_workspace.__dict__.keys():
        k_workspace.optimizer = k_workspace.model.get_optimizer(**k_cfg.optimizer)
    k_workspace.load_payload(k_payload, exclude_keys=None, include_keys=None)

    k_policy = k_workspace.model
    k_policy.to(device)
    k_policy.eval()
    print("keypose policy loaded")

    # load trajectory checkpoint
    t_payload = torch.load(open(trajectory_ckpt, 'rb'), pickle_module=dill)
    t_cfg = t_payload['cfg']

    cls = hydra.utils.get_class(t_cfg._target_)
    t_workspace = cls(t_cfg, output_dir=output)
    t_workspace: BaseWorkspace
    ## in case that model, ema_model & opt are not defined in __init__ (e.g. ddp)
    if "model" not in t_workspace.__dict__.keys():
        t_workspace.model = hydra.utils.instantiate(t_cfg.policy)
    if "ema_model" not in t_workspace.__dict__.keys() and t_cfg.training.use_ema:
        t_workspace.ema_model = copy.deepcopy(t_workspace.model)
    if "optimizer" not in t_workspace.__dict__.keys():
        t_workspace.optimizer = t_workspace.model.get_optimizer(**t_cfg.optimizer)
    t_workspace.load_payload(t_payload, exclude_keys=None, include_keys=None)

    if hasattr(t_workspace.model, "num_inference_steps"):
        t_policy = t_workspace.model
        if t_cfg.training.use_ema:
            t_policy = t_workspace.ema_model
        
        t_policy.to(device)
        t_policy.eval()
        print("trajectory policy loaded")

        ## set inference params
        t_policy.num_inference_steps = num_inference_steps
    else:
        raise RuntimeError("Unsupported policy type: ", t_cfg.name)
    
    policy = KeyposeTrajectoryPolicy(k_policy, t_policy, epsilon=epsilon)
    
    ##############################
    #      prepare env params    #
    ##############################
    shape_meta = t_cfg.task.shape_meta
    state_dim = shape_meta.obs.qpos.shape[0] ## qpos shape
    camera_names = [key for key in shape_meta.obs.keys() if key.startswith("cam")]
    c, h, w = shape_meta.obs.cam_high.shape ## [c, h, w]

    n_obs_steps = t_cfg.n_obs_steps
    print("n_obs_steps: ", n_obs_steps)
    print("max_timesteps:", max_timesteps)

    ## load aloha env
    env = make_real_env(init_node=True, downsample_scale=scale)

    ## rollout
    max_timesteps = int(max_timesteps) # T
    num_rollouts = 1

    ##############################
    #       rollout starts       #
    ##############################
    for rollout_idx in range(num_rollouts):
        rollout_idx += 1

        print(f"Rollout {rollout_idx} - Collecting observations...")

        ## reset env
        ts = env.reset()
        t_idx = pad_before = n_obs_steps - 1

        qpos_history = np.zeros(
            (max_timesteps + pad_before, state_dim), dtype=np.float32
        ) # [T+To-1, state_dim]
        images_history = dict()
        for cam_name in camera_names:
            images_history[cam_name] = np.zeros(
                (max_timesteps + pad_before, c, h, w), dtype=np.float32
            ) # [T+To-1, c, h, w]

        qpos_history, images_history = collect_obs(ts, t_idx, n_obs_steps, qpos_history, images_history, camera_names, shape_meta)
        ep_t0 = time.perf_counter()
        step_time_list = []
        with torch.inference_mode():
            ## loop max_timesteps
            while True:
                ''' construct observations_seq = {"images", "qpos"} '''
                # horizon indices: t-To+1, ..., t-1, t
                t_start = t_idx + 1 - n_obs_steps
                t_end = t_idx + 1

                obs_dict_np = dict()
                obs_dict_np["qpos"] = qpos_history[t_start:t_end] ## [n_obs_steps, state_dim]")
                for cam_name in camera_names:
                    obs_dict_np[cam_name] = images_history[cam_name][t_start:t_end] ## [n_obs_steps, c, h, w]

                print(f"observation_range = [{t_start}:{t_end})")

                ''' get action sequence '''
                t0 = time.perf_counter()
                obs_dict = dict_apply(obs_dict_np, 
                    lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                result = policy.predict_action(obs_dict)
                t1 = time.perf_counter()
                step_time_list.append(t1 - t0)
                print(f"Execution Policy: {t1 - t0:.4f} [s]")
                # ipdb.set_trace()

                action_seq = result['action'][0].detach().to('cpu').numpy() # (Ta,Da)

                ''' implement action sequence '''
                for action in action_seq:
                    ts = env.step(action) # ts_(t+1)
                    t_idx += 1
                    
                    if t_idx == max_timesteps + pad_before:
                        break

                    qpos_history, images_history = collect_obs(ts, t_idx, n_obs_steps, qpos_history, images_history, camera_names, shape_meta)
                
                if t_idx >= max_timesteps + pad_before:
                    break

    ### move grippers
    PUPPET_GRIPPER_JOINT_OPEN = 1.4910
    move_grippers([env.puppet_bot_left, env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)  # open
    ep_t1 = time.perf_counter()
    print(f"Average step time: {np.mean(step_time_list[1:]):.4f} +/- {np.std(step_time_list[1:]):.4f} s")
    print(f"Total time: {ep_t1 - ep_t0:.4f} s")

    ##############################
    #   save videos and stats    #
    ##############################
    n_cam = len(camera_names)
    rgb_seq = np.full((max_timesteps, n_cam, 3, h, w), np.nan, dtype=np.uint8)
    for i, cam_name in enumerate(camera_names):
        rgb_seq[:, i] = (images_history[cam_name][pad_before:] * 255).astype(np.uint8)
    rgb_seq = rearrange(rgb_seq, 't n c h w -> t h (n w) c')

    video_recorder = VideoRecorder.create_h264(
        fps=50,
        codec="h264",
        input_pix_fmt="rgb24",
        crf=22,
        thread_type="FRAME",
        thread_count=1,
    )
    video_recorder.stop()
    video_recorder.start(os.path.join(output, f"rollout_{rollout_idx}.mp4"))
    for rgb in rgb_seq:
        video_recorder.write_frame(rgb)
    video_recorder.stop()

    json_log = dict()
    json_log["ckpt_keypose"] = keypose_ckpt
    json_log["ckpt_trajectory"] = trajectory_ckpt
    json_log["epsilon"] = epsilon
    json_log["num_inference_steps"] = num_inference_steps
    output_path = os.path.join(output, 'eval_log.json')
    json.dump(json_log, open(output_path, 'w'), indent=2, sort_keys=True)
                        

def collect_obs(ts, idx, n_obs_steps, qpos_history, images_history, camera_names, shape_meta):
    obs = ts.observation
    ## get qpos input
    qpos = np.array(obs['qpos']) ## [state_dim,]
    if idx == n_obs_steps - 1:
        qpos_history[idx+1-n_obs_steps:idx+1] = qpos # broadcast here
    else:
        qpos_history[idx] = qpos

    ## get image input
    curr_image_dict = get_image(ts, camera_names, shape_meta) # it returns a dict
    if idx == n_obs_steps:
        for cam_name in camera_names:
            images_history[cam_name][idx+1-n_obs_steps:idx+1] = curr_image_dict[cam_name]
    else:
        for cam_name in camera_names:
            images_history[cam_name][idx] = curr_image_dict[cam_name]

    return qpos_history, images_history



def get_image(ts, camera_names, shape_meta):
    """
    @return
        curr_images_dict: {
            cam_name: image (c,h,w)
        }
    """
    curr_images_dict = dict()
    for cam_name in camera_names:
        # h,w,c --> c,h,w
        curr_image = np.moveaxis(ts.observation['images'][cam_name], -1, 0) / 255.0
        assert curr_image.shape == tuple((shape_meta.obs[cam_name]).shape), \
            f"{curr_image.shape} vs. {tuple((shape_meta.obs[cam_name]).shape)}"
        curr_images_dict[cam_name] = curr_image # [0, 1]^(c,h,w)

    return curr_images_dict



if __name__ == '__main__':
    main()
