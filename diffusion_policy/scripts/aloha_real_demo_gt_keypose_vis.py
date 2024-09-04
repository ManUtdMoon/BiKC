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

from diffusion_policy.gym_util.video_recording_wrapper import VideoRecorder


@click.command()
@click.option('--task', '-t', default="aloha_insert_10s_random_init")
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

    rgb_keys = ["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"]

    with tqdm(total=num_episodes, desc="Process", mininterval=1.0) as pbar:
        # for each demo in the dataset
        for i in range(num_episodes):
            dataset_path = os.path.join(dataset_dir, f"episode_{i}.hdf5")
            video_recorder.start(os.path.join(output_dir, f"episode_{i}.mp4"))
            with h5py.File(dataset_path, "r+") as demo:
                qpos = demo[f"/observations/qpos"][:].astype(np.float32)
                steps = np.arange(qpos.shape[0])

                for j in steps:
                    rgb = np.concatenate(
                        [demo[f"/observations/images/{key}"][j].astype(np.uint8) for key in rgb_keys], axis=1
                    )

                    video_recorder.write_frame(rgb)
                video_recorder.stop()
                pbar.update(1)


if __name__ == "__main__":
    main()