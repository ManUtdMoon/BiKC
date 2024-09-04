import numpy as np
import gym
from gym import spaces
import cv2
from dm_control.rl.control import Environment
from dm_control.mujoco.engine import Camera
import modern_robotics as mr
from einops import rearrange

from diffusion_policy.env.aloha.constants import (
    vx300s, LEFT_BASE_POSE, RIGHT_BASE_POSE
)

class AlohaImageWrapper(gym.Env):
    metadata = {
        "render.modes": ["rgb_array"], 
        "video.frames_per_second": 10
    }

    def __init__(
        self,
        env: Environment,
        shape_meta: dict,
        render_obs_key: list = ["top", "angle", "front_close"],
    ):
        self.env = env
        self.render_obs_key = render_obs_key
        self._seed = None
        self.shape_meta = shape_meta
        self.render_cache = None
        self.target_keypose = None
        self.has_reset_before = False

        # setup spaces
        action_shape = shape_meta["action"]["shape"]
        action_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=action_shape,
            dtype=np.float32,
        )
        self.action_space = action_space

        observation_space = spaces.Dict({
            "qpos": spaces.Box(-np.inf, np.inf, shape=(14,), dtype=np.float32),
            "images": spaces.Box(0.0, 1.0, shape=(3, 240, 320), dtype=np.float32),
        })
        self.observation_space = observation_space


    def get_observation(self, raw_obs=None):
        if raw_obs is None:
            raw_obs = self.env._task.get_observation(self.env._physics)

        render_cache = []
        for key in self.render_obs_key:
            if key not in raw_obs["images"].keys():
                raise ValueError(f"key {key} not in raw_obs['images']")
            render_cache.append(raw_obs["images"][key]) # [(h, w, c)]
        # concat and reshape into (h, w*3, c)
        self.render_cache = render_cache

        # raw obs is a dict, we need to extract key-val like shape_meta
        obs = dict()
        for key in self.shape_meta["obs"].keys():
            if key.endswith("images"):
                # resize
                h, w = self.shape_meta["obs"][key]["shape"][1:]
                img = cv2.resize(raw_obs["images"]["top"], (w, h),  interpolation=cv2.INTER_AREA)
                # h, w, c --> c, h, w
                # [0, 255] --> [0, 1]
                obs[key] = np.moveaxis(img.astype(np.float32) / 255.0, -1, 0)
            else:
                obs[key] = raw_obs[key]
        return obs

    def seed(self, seed=None):
        assert isinstance(self.env._task._random, np.random.RandomState)
        self.env._task._random = np.random.RandomState(seed)
        self._seed = seed

    def reset(self):
        ts = self.env.reset()
        self.target_keypose = None
        return self.get_observation(ts.observation)

    def step(self, action):
        """
        action: concat of action & target_keypose
        """
        Da = self.shape_meta["action"]["shape"][0]
        act_seq, target = np.split(action, [Da], axis=-1)
        self.target_keypose = target

        ts = self.env.step(act_seq)

        raw_obs = ts.observation
        reward = ts.reward
        done = ts.last()
        info = dict()

        obs = self.get_observation(raw_obs)
        return obs, reward, done, info

    def render(self, mode="rgb_array"):
        if self.render_cache is None:
            raise RuntimeError('Must run reset or step before render.')
        # render_cache is a list of (h, w, c) uint8 array
        
        ### edit on image to plot keypose
        # from qpos to pixel
        ee_pix = _qpos_to_ee_pix(
            self.target_keypose,
            self.env._physics,
            cam_id_list=self.render_obs_key
        )

        # draw keypose on 2d img
        for rgb, pix_pos in zip(self.render_cache, ee_pix):
            lu, lv = pix_pos[0]
            ru, rv = pix_pos[1]
            
            rgb[lv-5:lv+5, lu-5:lu+5] = [0, 255, 0]
            rgb[rv-5:rv+5, ru-5:ru+5] = [255, 127, 14]
            
        img = np.concatenate(self.render_cache, axis=1)

        return img


def _qpos_to_ee_pix(
    qpos,
    physics,
    cam_id_list
):
    """
    @params
        qpos: np.ndarray
            14
        physics: mujoco.Physics
        cam_id_list: [cam_id x ncam] to render
    @return
        ee_pix: list of np.ndarray(2), len = ncam
    """
    ## from qpos to ee in world coordinate
    assert qpos.ndim == 1
    left, right = np.split(qpos, 2)
    left_qpos, right_qpos = left[:-1], right[:-1]
    
    expected_shape = (4, 4)
    
    left_ee_lb = mr.FKinSpace(vx300s.M, vx300s.Slist, left_qpos)
    right_ee_rb = mr.FKinSpace(vx300s.M, vx300s.Slist, right_qpos)

    left_ee_w = np.matmul(LEFT_BASE_POSE, left_ee_lb)
    right_ee_w = np.matmul(RIGHT_BASE_POSE, right_ee_rb)

    assert left_ee_w.shape == expected_shape    

    ee_w = np.concatenate([left_ee_w[:, -1:], right_ee_w[:, -1:]], axis=1) # (4, 2)

    ## from ee_w to ee_pix
    cam_mats = np.asarray([
        Camera(physics, 480, 640, camera_id=cam_id).matrix
        for cam_id in cam_id_list
    ]) # (ncam, 3, 4)

    ee_pix_homo = cam_mats @ ee_w # (ncam, 3, 2)
    ee_pix_xy, ee_pix_s = ee_pix_homo[:, :2],  ee_pix_homo[:, -1:]
    ee_pix = np.round(ee_pix_xy / ee_pix_s).astype(int) # (n,2,2)

    return np.swapaxes(ee_pix, 1, 2) # (n,xy,lr) -> (n,lr,xy)
