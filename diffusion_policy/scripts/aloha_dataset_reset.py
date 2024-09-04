if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)

import os
import click
import pathlib
import h5py
from tqdm import tqdm


@click.command()
@click.option('--task', '-t',  required=True)
@click.option('--num_episodes', '-n', default=1, type=int)
def main(task: str, num_episodes: int):
    proj_dir = pathlib.Path(__file__).parent.parent.parent
    dataset_dir = proj_dir / "data/aloha/datasets" / task
    dataset_dir = str(dataset_dir.expanduser())

    init_keys = [
        "observations",
        "action",
    ]

    init_obs_keys = [
        "images",
        "qpos",
        "qvel"
    ]

    for i in range(num_episodes):
        dataset_file = os.path.join(dataset_dir, f"episode_{i}.hdf5")
        with h5py.File(dataset_file, "r+") as demo:
            print(f"Episode {i:>3d}: {demo['keyposes/merge'][()]}.")

            for key in demo.keys():
                if key not in init_keys:
                    del demo[key]

                if key == "observations":
                    for subkey in demo[key].keys():
                        if subkey not in init_obs_keys:
                            del demo[f"{key}/{subkey}"]
            
            # check
            for key in demo.keys():
                assert key in init_keys, f"key {key} not in init keys {init_keys}"
                if key == "observations":
                    for subkey in demo[key].keys():
                        assert subkey in init_obs_keys, f"subkey {subkey} not in init obs keys {init_obs_keys}"

            if i == 0:
                demo.visit(
                    lambda name: print(name, demo[name].shape) \
                        if isinstance(demo[name], h5py.Dataset) \
                        else None
                )


if __name__ == "__main__":
    main()