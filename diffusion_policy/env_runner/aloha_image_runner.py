import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import dill
import math
import wandb.sdk.data_types.video as wv
import gym
import gym.spaces as spaces
from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import (
    VideoRecordingWrapper,
    VideoRecorder,
)

from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.env.aloha.aloha_image_wrapper import AlohaImageWrapper
from diffusion_policy.env.aloha.sim_env import make_sim_env, BOX_POSE
from diffusion_policy.env.aloha.act_utils import sample_box_pose, sample_insertion_pose


class AlohaImageRunner(BaseImageRunner):
    def __init__(
        self,
        output_dir,
        task_name,
        shape_meta: dict,
        n_train=0,
        n_train_vis=0,
        train_start_seed=0,
        n_test=30,
        n_test_vis=6,
        test_start_seed=10000,
        max_steps=400,
        n_obs_steps=4,
        n_action_steps=8,
        fps=10,
        crf=22,
        past_action=False,
        tqdm_interval_sec=5.0,
        n_envs=None,
        multiplier=1,
    ):
        super().__init__(output_dir)
        if n_envs is None:
            n_envs = n_train + n_test

        steps_per_render = max(10 // fps, 1)

        def env_fn():
            env = make_sim_env(task_name)
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    AlohaImageWrapper(
                        env=env,
                        shape_meta=shape_meta
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec="h264",
                        input_pix_fmt="rgb24",
                        crf=crf,
                        thread_type="FRAME",
                        thread_count=1,
                    ),
                    file_path=None,
                    steps_per_render=steps_per_render,
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
            )

        env_fns = [env_fn] * n_envs
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()
        # train
        for i in range(n_train):
            seed = train_start_seed + i
            enable_render = i < n_train_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        "media", f"train_{seed}" + ".mp4"
                    )
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # set seed
                assert isinstance(env, MultiStepWrapper)
                env.seed(seed)

            env_seeds.append(seed)
            env_prefixs.append("train/")
            env_init_fn_dills.append(dill.dumps(init_fn))

        # test
        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        "media", f"test_{seed}" + ".mp4"
                    )
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # set seed
                assert isinstance(env, MultiStepWrapper)
                env.seed(seed)

            env_seeds.append(seed)
            env_prefixs.append("test/")
            env_init_fn_dills.append(dill.dumps(init_fn))

        def dummy_env_fn():
            # Avoid importing or using env in the main process
            # to prevent OpenGL context issue with fork.
            # Create a fake env whose sole purpos is to provide 
            # obs/action spaces and metadata.
            env = gym.Env()
            env.observation_space = spaces.Dict({
                "qpos": spaces.Box(-np.inf, np.inf, shape=(14,), dtype=np.float32),
                "images": spaces.Box(0.0, 1.0, shape=(3, 480, 640), dtype=np.float32),
            })
            env.action_space = gym.spaces.Box(
                -np.inf, np.inf, shape=(14,), dtype=np.float32
            )
            env.metadata = {
                'render.modes': ['rgb_array'],
                'video.frames_per_second': 10
            }
            env = MultiStepWrapper(
                env=env,
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps
            )
            return env

        env = SyncVectorEnv(env_fns)  #, dummy_env_fn=dummy_env_fn)

        # test env
        # env.reset(seed=env_seeds)
        # x = env.step(env.action_space.sample())
        # imgs = env.call('render')
        # import pdb; pdb.set_trace()

        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec
        self.task_name = task_name

    def run(self, policy: BaseImagePolicy):
        device = policy.device
        dtype = policy.dtype
        env = self.env

        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0, this_n_active_envs)

            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]] * n_diff)
            assert len(this_init_fns) == n_envs

            # init envs
            env.call_each("run_dill_function", args_list=[(x,) for x in this_init_fns])

            # start rollout
            ### set task
            if "sim_transfer_cube" in self.task_name:
                BOX_POSE[:200] = [
                    sample_box_pose() for _ in range(200)
                ]  # used in sim reset
            elif "sim_insertion" in self.task_name:
                BOX_POSE[:200] = [
                    np.concatenate(sample_insertion_pose()) for _ in range(200)
                ]  # used in sim reset
            obs = env.reset()
            past_action = None
            policy.reset()

            pbar = tqdm.tqdm(
                total=self.max_steps,
                desc=f"Eval AlohaImageRunner {chunk_idx+1}/{n_chunks}",
                leave=False,
                mininterval=self.tqdm_interval_sec,
            )
            done = False
            while not done:
                # create obs dict
                np_obs_dict = dict(obs)
                if self.past_action and (past_action is not None):
                    # TODO: not tested
                    np_obs_dict["past_action"] = past_action[
                        :, -(self.n_obs_steps - 1) :
                    ].astype(np.float32)

                # device transfer
                obs_dict = dict_apply(
                    np_obs_dict, lambda x: torch.from_numpy(x).to(device=device)
                )

                # run policy
                with torch.inference_mode():
                    action_dict = policy.predict_action(obs_dict)

                # device_transfer
                np_action_dict = dict_apply(
                    action_dict, lambda x: x.detach().to("cpu").numpy()
                )
                
                # concat action (B,Ta,Da) and keypose (B,Da) -> (B,Ta,Da*2)
                act_seq = np_action_dict["action"]
                target_keypose = np_action_dict["target_keypose"]
                Ta = act_seq.shape[1]
                target_keypose = np.tile(target_keypose[:, None, :], (1, Ta, 1))
                action = np.concatenate([act_seq, target_keypose], axis=-1)

                # step env
                obs, reward, done, info = env.step(action)
                done = np.all(done)
                past_action = action

                # update pbar
                pbar.update(action.shape[1])
            pbar.close()

            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call("get_attr", "reward")[
                this_local_slice
            ]
        # clear out video buffer
        _ = env.reset()

        # log
        max_rewards = collections.defaultdict(list)
        log_data = dict()
        for i in range(len(self.env_fns)):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            log_data[prefix + f"sim_max_reward_{seed}"] = max_reward

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix + f"sim_video_{seed}"] = sim_video

        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix + "mean_score"
            value = np.mean(value)
            log_data[name] = value

        # calculate and log the numbers of finishing different stages
        # i.e., rewards >= 2.0, 3.0, 4.0
        for prefix, value in max_rewards.items():
            v = np.array(value)
            log_data[prefix + "1st"] = np.around(np.sum(v > 1.5) / n_envs * 100, 2)
            log_data[prefix + "2nd"] = np.around(np.sum(v > 2.5) / n_envs * 100, 2)
            log_data[prefix + "3rd"] = np.around(np.sum(v > 3.5) / n_envs * 100, 2)

        return log_data
