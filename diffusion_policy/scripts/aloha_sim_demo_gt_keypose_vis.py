if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)


import os
import click
import pathlib
import numpy as np
import modern_robotics as mr
import h5py
from tqdm import tqdm


from diffusion_policy.env.aloha.constants import (
    vx300s, LEFT_BASE_POSE, RIGHT_BASE_POSE,
)
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecorder


@click.command()
@click.option('--task', '-t', default="sim_insertion_scripted")
@click.option('--num_episodes', '-n', default=1, type=int)
def main(task, num_episodes):
    proj_dir = pathlib.Path(__file__).parent.parent.parent.expanduser()

    # load dataset
    dataset_dir = proj_dir / "data/aloha/datasets" / task
    output_dir = str(dataset_dir / "gt_keypose")
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    video_recorder = VideoRecorder.create_h264(
        fps=50,
        codec="h264",
        input_pix_fmt="rgb24",
        crf=22,
        thread_type="FRAME",
        thread_count=1,
    )
    video_recorder.stop()

    with tqdm(total=num_episodes, desc="Process", mininterval=1.0) as pbar:
        # for each demo in the dataset
        for i in range(num_episodes):
            dataset_path = os.path.join(dataset_dir, f"episode_{i}.hdf5")
            video_recorder.start(os.path.join(output_dir, f"episode_{i}.mp4"))
            with h5py.File(dataset_path, "r+") as demo:
                imgs = demo[f'/observations/images/top'][:].astype(np.uint8)
                qpos = demo[f'/observations/qpos'][:].astype(np.float32)
                keyposes_inds = demo[f'/keyposes/merge'][:].astype(np.int16)

                steps = np.arange(qpos.shape[0])
                next_keypose_idx = np.searchsorted(keyposes_inds, steps, side='right')
                next_keypose_idx[-1] = next_keypose_idx[-2]
                next_keypose_idx = keyposes_inds[next_keypose_idx]
                for j, img in enumerate(imgs):
                    next_keypose = qpos[next_keypose_idx[j]]

                    pix_pos = _qpos_to_ee_pix(next_keypose)
                    
                    # turn qpos keypose to eepos in pixel frame
                    # draw keypose on 2d img
                    rgb = img.copy()
                    lu, lv = pix_pos[0]
                    ru, rv = pix_pos[1]
                    
                    rgb[lv-2:lv+2, lu-2:lu+2] = [0, 255, 0]
                    rgb[rv-2:rv+2, ru-2:ru+2] = [255, 127, 14]

                    video_recorder.write_frame(rgb)
                video_recorder.stop()
                pbar.update(1)


def _qpos_to_ee_pix(qpos):
    """
    @params
        qpos: np.ndarray
            14
    @return
        ee_pix: np.ndarray(2, 2)
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
    cam_mat = np.array(
        [
            [-296.37531757, 0.0, 319.5, -255.6 ],
            [   0.0, 296.37531757, 239.5, -369.42519054],
            [   0.0,  0.0, 1., -0.8],
        ],
        dtype=np.float32
    ) # (3, 4)

    ee_pix_homo = cam_mat @ ee_w # (3, 2)
    ee_pix_xy, ee_pix_s = ee_pix_homo[:2],  ee_pix_homo[-1:]
    ee_pix = np.round(ee_pix_xy / ee_pix_s / 2).astype(int) # (2,2), divided by 2 because img is downsampled

    return np.swapaxes(ee_pix, 0, 1) # (xy,lr) -> (lr,xy)


if __name__ == "__main__":
    main()