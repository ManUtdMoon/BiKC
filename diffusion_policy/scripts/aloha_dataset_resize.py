if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)


import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import cv2
import click
import h5py
from tqdm import tqdm


@click.command()
@click.option('--task', '-t',  required=True)
@click.option('--num_episodes', '-n', default=1, type=int)
@click.option('--scale', '-s', default=4, type=int)
def main(task, num_episodes, scale):
    proj_dir = pathlib.Path(__file__).parent.parent.parent
    dataset_dir = proj_dir / "data/aloha/datasets" / task / "original"
    dataset_dir = str(pathlib.Path(dataset_dir).expanduser())

    output_dir = proj_dir / "data/aloha/datasets" / task
    output_dir = str(pathlib.Path(output_dir).expanduser())

    original_shape = None
    resized_shape = None

    with tqdm(total=num_episodes, desc="Process", mininterval=1.0) as pbar:
        for i in range(num_episodes):
            dataset_path = os.path.join(dataset_dir, f"episode_{i}.hdf5")
            output_path = os.path.join(output_dir, f"episode_{i}.hdf5")

            with h5py.File(dataset_path, "r") as src:
                with h5py.File(output_path, "w", rdcc_nbytes=1024**2*2) as dst:
                    ### step 1: action, direct copy
                    src.copy(src["action"], dst, "action")

                    ### step 2: non-images observations
                    obs = dst.create_group("observations")
                    for key in src["observations/"].keys():
                        if "images" not in key:

                            src.copy(src["observations/"+key], dst["observations/"], key)
                    
                    ### step 3: images observations
                    # get image size
                    if original_shape is None:
                        example_key = list(src["observations/images/"].keys())[0]
                        original_shape = src["observations/images/"+example_key][0].shape
                        h, w, c = original_shape
                        resized_shape = (h//scale, w//scale, c)
                        print(f"Original: {original_shape}, Resized: {resized_shape}")
                    
                    image = obs.create_group("images")
                    h, w, c = resized_shape
                    for cam_name in src["observations/images/"].keys():
                        time_steps = src["observations/images/"+cam_name].shape[0]
                        _ = image.create_dataset(
                            cam_name, (time_steps, h, w, c), dtype=np.uint8,
                            chunks=(1, h, w, c)
                        )
                        for t in range(time_steps):
                            frame = src["observations/images/"+cam_name][t]
                            resized_frame = cv2.resize(frame, (w, h), cv2.INTER_AREA)
                            dst["observations/images/"+cam_name][t] = resized_frame
                            if t == 0 and i == 0:
                                # save the two images
                                fig, ax = plt.subplots(1, 2)
                                ax[0].imshow(frame)
                                ax[1].imshow(resized_frame)
                                plt.savefig(f"{output_dir}/{t}_{cam_name}.png", dpi=300)

                    if i == 0:
                        dst.visit(
                            lambda name: print(name, dst[name].shape) \
                                if isinstance(dst[name], h5py.Dataset) \
                                else None
                        )

            pbar.update(1)

if __name__ == "__main__":
    main()