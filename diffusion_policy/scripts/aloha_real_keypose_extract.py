if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)

from typing import List
import os
import click
import pathlib
import cv2
import numpy as np
import matplotlib.pyplot as plt
import modern_robotics as mr
import h5py
from tqdm import tqdm

from diffusion_policy.env.aloha.constants import (
    DT, vx300s, REAL_LEFT_BASE_POSE, REAL_RIGHT_BASE_POSE,
    REAL_GRIPPER_EPSILON, REAL_EE_DIST_BOUND
)


def _smooth(data, window_size=5):
    if window_size % 2 == 0:
        print("window size must be odd, add 1 autonomously.")
        window_size += 1
    data = np.pad(data, (window_size // 2, window_size // 2), mode='edge')
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


def _load_trajectory(dataset_dir: str, i: int):
    '''load h5df trajectory and return dict of sequences of interest
    params:
        dataset_dir: str
        i: int, episode index
    return:
        dict of sequences of interest
    '''
    dataset_path = os.path.join(dataset_dir, f"episode_{i}.hdf5")
    with h5py.File(dataset_path, "r") as demo:
        ### load images and save videos
        this_image = dict()
        for cam_name in demo[f'/observations/images/'].keys():
            this_image[cam_name] = demo[f'/observations/images/{cam_name}'][:].astype(np.uint8)
        # _save_videos(this_image, DT, video_path=f'{dataset_dir}/episode_{i}.mp4')

        # extract qpos and gripper pos
        this_qpos_left = demo["observations/qpos"][:, :6].astype(np.float32)
        this_qpos_right = demo["observations/qpos"][:, 6+1:6+7].astype(np.float32)
        this_gripper_left = demo["observations/qpos"][:, 6].astype(np.float32)
        this_gripper_right = demo["observations/qpos"][:, 13].astype(np.float32)

        this_gripper_act_left = demo["action"][:, 6].astype(np.float32)
        this_gripper_act_right = demo["action"][:, 13].astype(np.float32)

        T = this_qpos_left.shape[0]
        ### extract EE information
        this_ee_pos_left = np.zeros((T, 3))
        this_ee_pos_right = np.zeros((T, 3))
        for j in range(T):
            # pos
            left_pose_mat = mr.FKinSpace(vx300s.M, vx300s.Slist, this_qpos_left[j])
            right_pose_mat = mr.FKinSpace(vx300s.M, vx300s.Slist, this_qpos_right[j])
            this_ee_pos_left[j] = np.dot(REAL_LEFT_BASE_POSE, left_pose_mat)[:3, 3]
            this_ee_pos_right[j] = np.dot(REAL_RIGHT_BASE_POSE, right_pose_mat)[:3, 3]
        
        # smooth before further computation: pos, gripper
        this_gripper_left = _smooth(this_gripper_left)
        this_gripper_right = _smooth(this_gripper_right)
        for i in range(3):
            this_ee_pos_left[:, i] = _smooth(this_ee_pos_left[:, i])
            this_ee_pos_right[:, i] = _smooth(this_ee_pos_right[:, i])

        this_ee_dpos_left = np.diff(this_ee_pos_left, axis=0) / DT
        this_ee_dpos_right = np.diff(this_ee_pos_right, axis=0) / DT

        this_ee_vel_norm_left = np.linalg.norm(this_ee_dpos_left, axis=-1)
        this_ee_vel_norm_right = np.linalg.norm(this_ee_dpos_right, axis=-1)

        this_ee_dist = np.linalg.norm(this_ee_pos_left - this_ee_pos_right, axis=-1)
        this_ee_ddist = np.diff(this_ee_dist) / DT

        this_trajectory = dict(
            qpos_left=this_qpos_left,
            qpos_right=this_qpos_right,
            gripper_left=this_gripper_left,
            gripper_right=this_gripper_right,
            gripper_act_left=this_gripper_act_left,
            gripper_act_right=this_gripper_act_right,
            ee_pos_left=this_ee_pos_left,
            ee_pos_right=this_ee_pos_right,
            ee_dpos_left=this_ee_dpos_left,
            ee_dpos_right=this_ee_dpos_right,
            ee_vel_norm_left=this_ee_vel_norm_left,
            ee_vel_norm_right=this_ee_vel_norm_right,
            ee_dist=this_ee_dist,
            ee_ddist=this_ee_ddist,
            image=this_image,
        )

        return this_trajectory


def _find_keypose_idx(
    task: str,
    **kwargs
):
    if task == "aloha_insert_10s_random_init":
        return _find_keypose_idx_insert(**kwargs)
    elif task == "aloha_screwdriver":
        return _find_keypose_idx_screwdriver(**kwargs)
    elif task == "aloha_starbucks":
        return _find_keypose_idx_starbucks(**kwargs)
    elif task == "aloha_conveyor" or task == "aloha_conveyor_old":
        return _find_keypose_idx_conveyor(**kwargs)
    else:
        raise NotImplementedError(f"task {task} not implemented.")


def _find_keypose_idx_insert(
    trajectory: dict,
    side: str="left",
    gripper_epsilon=REAL_GRIPPER_EPSILON,
):
    # return None, None, False
    # load data and initialization
    gripper = trajectory[f"gripper_{side}"]
    gripper_act = trajectory[f"gripper_act_{side}"]
    gripper_change_rate = np.diff(gripper) / DT
    ee_dist = trajectory["ee_dist"]
    T = len(gripper)
    keypose_indices = [0, T-1]  # init and final state are keyposes

    problem = False
    coordination = 0
    for i in range(100, 200):
        if (
            gripper_change_rate[i-2] < -gripper_epsilon and
            gripper_change_rate[i-1] < -gripper_epsilon and
            gripper_change_rate[i] >= -gripper_epsilon
        ):
            keypose_indices.append(i) # from closing to closed
            break
    for i in range(150, 400):
        if (
            ee_dist[i-2] > REAL_EE_DIST_BOUND + 0.02 and
            ee_dist[i-1] > REAL_EE_DIST_BOUND + 0.02 and
            ee_dist[i] <= REAL_EE_DIST_BOUND + 0.02
        ):
            keypose_indices.append(i) # from far to close
            coordination = i
            break
    for i in range(250, T-1):
        if (
            -gripper_epsilon < gripper_change_rate[i-2] < gripper_epsilon and
            -gripper_epsilon < gripper_change_rate[i-1] < gripper_epsilon and
            gripper_change_rate[i] >= gripper_epsilon
        ):
            keypose_indices.append(i) # from closed to opening

    return keypose_indices, coordination, problem


def _find_keypose_idx_screwdriver(
    trajectory: dict,
    side: str="left",
    gripper_epsilon=REAL_GRIPPER_EPSILON,
    z_thres=0.06,
):
    # return None, None, False
    gripper = trajectory[f"gripper_{side}"]
    gripper_z = trajectory[f"ee_pos_{side}"][:, 2]
    gripper_act = trajectory[f"gripper_act_{side}"]
    gripper_change_rate = np.diff(gripper) / DT

    T = len(gripper)  # 800
    keypose_indices = [0, T-1]  # init and final state are keyposes

    problem = False
    coordination = 0
    for i in range(100, 250):
        if (
            side == "left" and
            gripper_act[i-2] > gripper_epsilon and
            gripper_act[i-1] > gripper_epsilon and
            gripper_act[i] <= gripper_epsilon
        ):
            keypose_indices.append(i)  # left grasp the screwdriver
            break
    # for i in range(150, 300):
    #     if (
    #         side == "right" and
    #         gripper_z[i-2] > z_thres and
    #         gripper_z[i-1] > z_thres and
    #         gripper_z[i] <= z_thres
    #     ):
    #         keypose_indices.append(i)  # right press the box
    #         break
    # for i in range(250, 450):
    #     if (
    #         side == "left" and
    #         gripper_act[i-2] < gripper_epsilon and
    #         gripper_act[i-1] < gripper_epsilon and
    #         gripper_act[i] >= gripper_epsilon
    #     ):
    #         keypose_indices.append(i)  # left release the screwdriver
    #         break
    # for i in range(350, 500):
    #     if (
    #         side == "left" and
    #         gripper_z[i-2] > z_thres and
    #         gripper_z[i-1] > z_thres and
    #         gripper_z[i] <= z_thres
    #     ):
    #         # ep 0-39: 350-500
    #         # ep 40: 450-500
    #         # ep 41-49: 375-500
    #         keypose_indices.append(i)  # left press the screwdriver
    #         break
    # for i in range(700, 500, -1):
    #     if (
    #         side == "right" and
    #         gripper_z[i-2] < z_thres and
    #         gripper_z[i-1] < z_thres and
    #         gripper_z[i] >= z_thres
    #     ):
    #         keypose_indices.append(i)  # right close the box
    #         break
    for i in range(650, T-1):
        if (
            side == "right" and
            gripper_z[i-2] > z_thres and
            gripper_z[i-1] > z_thres and
            gripper_z[i] <= z_thres
        ):
            keypose_indices.append(i) # right tighten the box
            coordination = i
            break

    return keypose_indices, coordination, problem


def _find_keypose_idx_starbucks(
    trajectory: dict,
    side: str="left",
    gripper_epsilon=REAL_GRIPPER_EPSILON,
):
    # return None, None, False
    gripper = trajectory[f"gripper_{side}"]
    gripper_act = trajectory[f"gripper_act_{side}"]
    gripper_change_rate = np.diff(gripper) / DT
    ee_dist = trajectory["ee_dist"]
    T = len(gripper)
    keypose_indices = [0, T-1]  # init and final state are keyposes

    problem = False
    coordination = T - 1

    return keypose_indices, coordination, problem


def _find_keypose_idx_conveyor(
    trajectory: dict,
    side: str="left",
    gripper_epsilon=REAL_GRIPPER_EPSILON,
):
    # return None, None, False
    gripper = trajectory[f"gripper_{side}"]
    gripper_act = trajectory[f"gripper_act_{side}"]
    gripper_change_rate = np.diff(gripper) / DT
    T = len(gripper)
    keypose_indices = [0, T-1]  # init and final state are keyposes

    problem = False
    coordination = 0

    for i in range(400, T-1):
        if (
            side == "right" and
            gripper_change_rate[i-2] < -gripper_epsilon and
            gripper_change_rate[i-1] < -gripper_epsilon and
            gripper_epsilon > gripper_change_rate[i] >= -gripper_epsilon
        ):
            keypose_indices.append(i) # from closing to closed
            break
    return keypose_indices, coordination, problem


def _plot_ee_and_gripper(
    trajectory: dict,
    keyposes: dict,
    dataset_dir: str,
    i: int,
):
    ### load data
    this_qpos_left = trajectory["qpos_left"]
    this_gripper_left = trajectory["gripper_left"]
    this_gripper_right = trajectory["gripper_right"]
    this_gripper_act_left = trajectory["gripper_act_left"]
    this_gripper_act_right = trajectory["gripper_act_right"]
    this_image = trajectory["image"]
    this_ee_pos_left = trajectory["ee_pos_left"]
    this_ee_pos_right = trajectory["ee_pos_right"]
    this_ee_dpos_left = trajectory["ee_dpos_left"]
    this_ee_dpos_right = trajectory["ee_dpos_right"]
    this_ee_vel_norm_left = trajectory["ee_vel_norm_left"]
    this_ee_vel_norm_right = trajectory["ee_vel_norm_right"]
    this_ee_dist = trajectory["ee_dist"]
    this_ee_ddist = trajectory["ee_ddist"]

    if keyposes is not None:
        keypose_left, coordination_left = keyposes["left"], keyposes["coordination_left"]
        keypose_right, coordination_right = keyposes["right"], keyposes["coordination_right"]
        merge = keyposes["merge"]

    ### pos of x, y, z & vel, gripper, abs_vel, ee dist and ee dist rate
    num_t, num_dim = this_qpos_left.shape[0], 3 + 3 + 1 + 1 + 2
    h, w = 2, num_dim
    num_figs = num_dim
    
    ### save images around keypose, assume there is only one cam
    cam_name = list(this_image.keys())[0]
    interval = 25
    step_around = lambda idx: np.clip(
        np.arange(idx-2*interval, idx+2*interval+1, interval),
        0, num_t - 1
    )
    
    ### create img dir if not exist
    img_dir = f'{dataset_dir}/imgs'
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    # steps_mat = list()
    # for idx in keypose_left:
    #     steps_mat.append(step_around(idx))
    # steps = np.stack(steps_mat, axis=0)  # (num_keypose, num_seq)
    # images = this_image[cam_name][steps]  # (num_keypose, num_seq, h, w, c)
    # images = rearrange(images, 'k t h w c -> (k h) (t w) c')
    # plt.imsave(f'{img_dir}/episode_{i}_left.png', images)

    # steps_mat = list()
    # for idx in keypose_right:
    #     steps_mat.append(step_around(idx))
    # steps = np.stack(steps_mat, axis=0)  # (num_keypose, num_seq)
    # images = this_image[cam_name][steps]  # (num_keypose, num_seq, h, w, c)
    # images = rearrange(images, 'k t h w c -> (k h) (t w) c')
    # plt.imsave(f'{img_dir}/episode_{i}_right.png', images)

    ### plot EE curves
    idx_ylabel_map = {
        0: r"$x$ [m]",
        1: r"$y$ [m]",
        2: r"$z$ [m]",
        3: r"$\dot{x}$ [m/s]",
        4: r"$\dot{y}$ [m/s]",
        5: r"$\dot{z}$ [m/s]",
        6: "gripper",
        7: r"$v_{\rm ee}$ [m/s]",
        8: r"$d_{\rm ee}$ [m]",
        9: r"$\dot{d}_{\rm ee} [m/s]$",
    }

    fig, axs = plt.subplots(num_figs, 1, figsize=(w, h * num_figs))
    t = np.arange(num_t) * DT
    for idx_dim in range(num_dim):
        ax = axs[idx_dim]
        if idx_dim < 3:
            ### x, y, z
            ax.plot(t, this_ee_pos_left[:, idx_dim], "r", label="left")
            ax.plot(t, this_ee_pos_right[:, idx_dim], "b", label="right")
            if idx_dim == 2:
                ax.plot(t, np.ones_like(t) * 0.06, 'k--')
            ax.legend()
        elif 3 <= idx_dim < 6:
            ### xdot, ydot, zdot
            ax.plot(t[:-1], this_ee_dpos_left[:, idx_dim-3], "r", label="left")
            ax.plot(t[:-1], this_ee_dpos_right[:, idx_dim-3], "b", label="right")
            ub = max(
                np.mean(this_ee_dpos_left[:, idx_dim-3]) + 1 * np.std(this_ee_dpos_left[:, idx_dim-3]),
                np.mean(this_ee_dpos_right[:, idx_dim-3]) + 1 * np.std(this_ee_dpos_right[:, idx_dim-3])
            )

            ax.legend()
            ax.axhline(y=0, color='k', linewidth=0.5)
            ax.set_ylim([-ub, ub])
        elif idx_dim == 6:
            ### gripper
            ax.plot(t, this_gripper_left, "r", label="left")
            ax.plot(t, this_gripper_right, "b", label="right")
            left_first_diff = np.diff(this_gripper_left) / DT
            right_first_diff = np.diff(this_gripper_right) / DT
            ax.plot(t[:-1], left_first_diff, "r--")
            ax.plot(t[:-1], right_first_diff, "b--")
            ax.plot(t, this_gripper_act_left, "r:")
            ax.plot(t, this_gripper_act_right, "b:")
            ax.plot(t, np.ones_like(t) * REAL_GRIPPER_EPSILON, 'k--')
            ax.plot(t, -np.ones_like(t) * REAL_GRIPPER_EPSILON, 'k--')

            if keyposes is not None:
                non_coordination_mask_left = np.array(keypose_left) != coordination_left
                non_coordination_mask_right = np.array(keypose_right) != coordination_right
                ax.scatter(
                    t[keypose_left][non_coordination_mask_left],
                    this_gripper_left[keypose_left][non_coordination_mask_left],
                    marker='x', color='r'
                )
                ax.scatter(
                    t[keypose_right][non_coordination_mask_right],
                    this_gripper_right[keypose_right][non_coordination_mask_right],
                    marker='x', color='b'
                )
                ax.scatter(
                    t[coordination_left], this_gripper_left[coordination_left],marker='o', color='r'
                )
                ax.scatter(
                    t[coordination_right], this_gripper_right[coordination_right], marker='o', color='b'
                )
                # plot a vertical line at merged keyposes
                for idx in merge:
                    ax.axvline(x=t[idx], color='k', linewidth=0.5)
            ax.set_ylim([-1.2, 1.2])
            ax.legend()
        elif idx_dim == 7:
            ### ee vel
            ax.plot(t[:-1], this_ee_vel_norm_left, "r", label="left")
            ax.plot(t[:-1], this_ee_vel_norm_right, "b", label="right")
            ax.plot(t, np.ones_like(t) * 0.05, 'k--')
            # set y limit
            ax.legend()
            ax.set_ylim([0, 0.1])
        elif idx_dim == 8:
            ### ee dist
            ax.plot(t, this_ee_dist, "r")
            ax.plot(t, np.ones_like(t) * REAL_EE_DIST_BOUND, 'k--')
        elif idx_dim == 9:
            ### ee dist rate
            ax.plot(t[:-1], this_ee_ddist, "b")
            ax.set_ylim([-1, 2])
        ax.set_xlabel("time [s]")
        ax.set_ylabel(idx_ylabel_map[idx_dim])

    plt.tight_layout()
    plt.savefig(f'{img_dir}/episode_{i}_ee.png', dpi=200)
    plt.close()


def _remove_too_close_keyposes(keypose_indices: List, min_dist=5, forward=False):
    if forward:
        # from the first keypose, keep the former one if
        # the distance is larger than min_dist
        refined_keypose_indices = [keypose_indices[0]]
        for idx in keypose_indices[1:]:
            if idx - refined_keypose_indices[-1] > min_dist:
                refined_keypose_indices.append(idx)
        if refined_keypose_indices[-1] != keypose_indices[-1]:
            refined_keypose_indices.append(keypose_indices[-1])
    else:
        # from the last keypose, keep the later one if 
        # the distance is larger than min_dist
        refined_keypose_indices = [keypose_indices[-1]]
        for idx in reversed(keypose_indices[:-1]):
            if refined_keypose_indices[-1] - idx > min_dist:
                refined_keypose_indices.append(idx)
        if refined_keypose_indices[-1] != 0:
            refined_keypose_indices.append(0)
        refined_keypose_indices = refined_keypose_indices[::-1]
    
    return np.array(refined_keypose_indices, dtype=np.int32)


@click.command()
@click.option('--task', '-t',  required=True)
@click.option('--start', '-s', default=0, type=int)
@click.option('--end', '-e', default=50, type=int)
@click.option('--dist', '-d', required=True, type=int, default=5)
@click.option('--forward', '-f', is_flag=True, default=False)
def main(task, start, end, dist, forward):
    proj_dir = pathlib.Path(__file__).parent.parent.parent
    dataset_dir = str((proj_dir / "data/aloha/datasets" / task).expanduser())
    for i in range(start, end):
        this_trajectory = _load_trajectory(dataset_dir, i)
        
        ### find keypose indices
        window_size = 5

        keypose_left, coordination_left, problem_left = _find_keypose_idx(
            task,
            trajectory=this_trajectory,
            side="left",
        )
        keypose_right, coordination_right, problem_right = _find_keypose_idx(
            task,
            trajectory=this_trajectory,
            side="right",
        )

        keyposes = None
        if keypose_left is not None:
            if problem_left:
                print(f'left problem in episode {i}')
            if problem_right:
                print(f'right problem in episode {i}')
            
            # merge left and right keyposes, seen as keyposes for 
            # the whole bimanual system
            keypose_left = np.asarray(keypose_left, dtype=np.int32)
            keypose_right = np.asarray(keypose_right, dtype=np.int32)
            assert keypose_left.ndim == keypose_right.ndim == 1
            merge = np.concatenate([keypose_left, keypose_right])
            merge = np.unique(merge)
            merge = np.sort(merge)
            merge = _remove_too_close_keyposes(merge, min_dist=dist, forward=forward)
            print(f"episode {i:>2d}, merge keyposes diff: {np.diff(merge)}")
            
            keyposes = dict(
                left=keypose_left,
                right=keypose_right,
                coordination_left=coordination_left,
                coordination_right=coordination_right,
                merge=merge,
            )
            
            ### load hdf5 and save keyposes into it
            dataset_path = os.path.join(dataset_dir, f"episode_{i}.hdf5")
            with h5py.File(dataset_path, "r+") as demo:
                ### first check if keyposes exists
                if "keyposes" in demo.keys():
                    ### warn that the keyposes will be overwritten
                    print(f"keyposes in episode {i} already exists, will be overwritten.")
                    del demo["keyposes"]

                keypose = demo.create_group("keyposes")
                keypose.create_dataset(
                    "merge", data=merge, dtype=np.int32
                )
                if i == 0:
                    demo.visit(lambda name: print(name))

        ### plot ee and gripper curves based on curves and calculated keyposes
        _plot_ee_and_gripper(
            this_trajectory, keyposes, dataset_dir, i
        )


if __name__ == '__main__':
    main()