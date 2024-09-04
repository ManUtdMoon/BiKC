from typing import Dict
from copy import deepcopy
import numpy as np
import torch
import modern_robotics as mr

from diffusion_policy.policy.keypose_base_policy import KeyposeBasePolicy
from diffusion_policy.policy.trajectory_base_policy import TrajectoryBasePolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env.aloha.constants import (
    DT, vx300s, LEFT_BASE_POSE, RIGHT_BASE_POSE
)


class KeyposeTrajectoryPolicy:
    ### init 
    # load two sub policies
    # maintain target_keypose
    def __init__(self,
        keypose_model: KeyposeBasePolicy,
        trajectory_model: TrajectoryBasePolicy,
        epsilon: float = 0.35,
    ):
        self.keypose_model = keypose_model
        self.trajectory_model = trajectory_model
        self.target_keypose = None
        self.last_keypose = None
        self.epsilon = epsilon # distance threshold for keypose matching

    def predict_action(self,
        obs_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        @params
            obs_dict: dict
                str: B,To,*
        @return
            result: dict
                action_pred: B,H,Da
                action: B,Ta,Da
                target_keypose: B,Da
        """
        k_nn = self.keypose_model
        t_nn = self.trajectory_model

        # current obs_dict: because obs_dict contains multi-step
        # o_(t-T_o+1), ..., o_(t-1), o_t,
        current_obs_dict = dict_apply(obs_dict, lambda x: x[:, -1].clone())
        current_pose = current_obs_dict["qpos"]

        ### option 1: update keyposes after reaching target, better for real-world
        # predict next keypose
        if self.target_keypose is None:
            # init step, all target_keypose should be updated
            self.last_keypose = current_obs_dict["qpos"].clone() # B,Da
            current_obs_dict["last_keypose"] = self.last_keypose
            self.target_keypose = k_nn.predict_next_keypose(current_obs_dict) # B,Da
            # print(f"target updated: {self.target_keypose}")
        
        # check if some current_poses are close to target_keyposes
        dist = dist_to_target(current_pose, self.target_keypose)
        print(f"dist: {dist}")

        reach_target = dist < self.epsilon # (B,)
        if reach_target.any():
            # update last_keypose and target_keypose
            self.last_keypose[reach_target] = self.target_keypose[reach_target]
            current_obs_dict["last_keypose"] = self.last_keypose
            self.target_keypose[reach_target] = k_nn.predict_next_keypose(current_obs_dict)[reach_target]
            # print(f"target updated: {self.target_keypose}")
        ### option 1 end

        ### option 2: update keypose at each step, better for sim
        # if self.target_keypose is None:
        #     self.last_keypose = current_obs_dict["qpos"].clone() # B,Da
        # else:
        #     dist = dist_to_target(current_pose, self.target_keypose)
        #     # print(f"dist: {dist}")
        #     reach_target = dist < self.epsilon
        #     if reach_target.any():
        #         self.last_keypose[reach_target] = self.target_keypose[reach_target]
        # current_obs_dict["last_keypose"] = self.last_keypose
        # self.target_keypose = k_nn.predict_next_keypose(current_obs_dict)
        ### option 2 end

        # predict sub trajectory
        result = t_nn.predict_trajectory(obs_dict, self.target_keypose)
        result["target_keypose"] = self.target_keypose.clone()

        return result
    
    # reset state for stateful policies
    def reset(self):
        self.keypose_model.reset()
        self.trajectory_model.reset()
        self.target_keypose = None
        self.last_keypose = None

    @property
    def device(self):
        return self.keypose_model.device
    
    @property
    def dtype(self):
        return self.keypose_model.dtype


def dist_to_target(
    x: torch.Tensor,
    target: torch.Tensor,
    dim: int = -1
):
    """
    @params
        x: torch.Tensor
            B,D
        target: torch.Tensor
            B,D
        dim: int

    @return
        d: torch.Tensor
            B
    """
    x = x.to(target.device)
    d = torch.linalg.vector_norm(x - target, dim=dim).to("cpu") # (B,)
    return d


def qpos_to_eepose(
    qpos: np.ndarray,
):
    """
    @params
        qpos: np.ndarray
            B, 14
    @return
        left_eepose: np.ndarray
            B, 4, 4
        right_eepose: np.ndarray
            B, 4, 4
    """
    if isinstance(qpos, torch.Tensor):
        qpos = qpos.cpu().numpy()
    left, right = np.split(qpos, 2, axis=-1)
    expected_shape = (qpos.shape[0], 4, 4)
    
    left_arm, right_arm = left[:, :-1], right[:, :-1]  # B, 6
    left_gripper, right_gripper = left[:, -1:], right[:, -1:] # B, 1
    relative_left_eepose = np.array(
        [mr.FKinSpace(vx300s.M, vx300s.Slist, this_qpos) for this_qpos in left_arm]
    )
    relative_right_eepose = np.array(
        [mr.FKinSpace(vx300s.M, vx300s.Slist, this_qpos) for this_qpos in right_arm]
    )
    left_eepose = np.matmul(LEFT_BASE_POSE, relative_left_eepose)
    right_eepose = np.matmul(RIGHT_BASE_POSE, relative_right_eepose)

    assert left_eepose.shape == expected_shape
    
    return left_eepose, right_eepose