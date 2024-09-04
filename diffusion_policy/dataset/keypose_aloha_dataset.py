if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)


import os
import h5py
from typing import Dict, List
import torch
import numpy as np
import copy
from tqdm import tqdm
import os
import shutil
from filelock import FileLock
from threadpoolctl import threadpool_limits
from einops import rearrange

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.sampler import get_val_mask
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.keypose_base_dataset import KeyposeBaseDataset
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
from diffusion_policy.common.normalize_util import get_image_range_normalizer


register_codecs()


class KeyposeAlohaDataset(KeyposeBaseDataset):
    def __init__(
        self,
        dataset_dir: str,
        shape_meta: dict,
        num_episodes=50,
        camera_names=["top"],
        seed=42,
        val_ratio=0.0,
        task="sim_transfer_cube_scripted",
        use_cache=False,
    ):
        super().__init__()
        '''
            structure of self.data:
            {
                "images": np.array([T, H, W, C]),
                "qpos": np.array([T, 14]), # a.k.a. current keypose
                "last_keypose": np.array([T, 14]),
                "next_keypose": np.array([T, 14]),
            }
        '''
        dataset_dir = os.path.expanduser(dataset_dir + "/" + task)
        if use_cache:
            cache_hdf5_path = os.path.join(dataset_dir, "keypose_cache.hdf5")
            cache_lock_path = cache_hdf5_path + ".lock"
            print("Acquiring lock on cache.")
            with FileLock(cache_lock_path):
                if not os.path.exists(cache_hdf5_path):
                    try:
                        print("Cache does not exsit. Creating!")
                        data = self.load_episodes_to_data(
                            num_episodes=num_episodes,
                            dataset_dir=dataset_dir,
                            camera_names=camera_names,
                        )
                        print("Saving cache to disk.")
                        with h5py.File(cache_hdf5_path, 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
                            root.attrs["sim"] = True
                            for key, value in data.items():
                                root.create_dataset(key, data=value)
                    except Exception as e:
                        shutil.rmtree(cache_hdf5_path)
                        raise e
                else:
                    print("Loading cached keypose data from disk.")
                    with h5py.File(cache_hdf5_path, 'r') as root:
                        data = dict()
                        for key in root.keys():
                            data[key] = root[key][()]
                    print("Loaded!")
        else:
            data = self.load_episodes_to_data(
                num_episodes=num_episodes,
                dataset_dir=dataset_dir,
                camera_names=camera_names,
            )

        rgb_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            type = attr.get("type", "low_dim")
            if type == "rgb":
                rgb_keys.append(key)
            elif type == "low_dim":
                lowdim_keys.append(key)

        val_mask = get_val_mask(
            n_episodes=len(next(iter(data.values()))),
            val_ratio=val_ratio,
            seed=seed,
        )
        train_mask = ~val_mask
        
        self.data = data
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.train_mask = train_mask
        self.train_indices = np.where(train_mask)[0]
        self.camera_names = camera_names

    def load_episodes_to_data(
        self,
        num_episodes,
        dataset_dir,
        camera_names,
    ):
        data = _create_empty_data()
        for i in tqdm(range(num_episodes)):  # num_episodes
            dataset_path = os.path.join(dataset_dir, f"episode_{i}.hdf5")
            with h5py.File(dataset_path, "r") as root:
                qpos = root["/observations/qpos"][()]
                keypose_idx = root["/keyposes/merge"][()]
                # next_keypose_idx (the last one is itself)
                next_keypose_idx = np.concatenate(
                    [keypose_idx[1:], keypose_idx[-1:]]
                )
                # last_keypose_idx (the first one is itself)
                last_keypose_idx = np.concatenate(
                    [keypose_idx[:1], keypose_idx[:-1]]
                )
                
                step = np.arange(len(qpos))
                assert step.size == 400
                next_keypose_idx = np.searchsorted(
                    keypose_idx, step, side='right')
                next_keypose_idx[-1] = next_keypose_idx[-2]
                next_keypose_idx = keypose_idx[next_keypose_idx]
                last_keypose_idx = np.searchsorted(
                    keypose_idx, step, side='left') - 1
                last_keypose_idx[0] = 0
                last_keypose_idx = keypose_idx[last_keypose_idx]
                
                # # print test
                # print("     keypose_idx:", keypose_idx)
                # print("last_keypose_idx:", last_keypose_idx[:20:2])
                # print("            step:", step[:20:2])
                # print("next_keypose_idx:", next_keypose_idx[:20:2])
                # new axis for different cameras
                all_cam_images = []
                for cam_name in camera_names:
                    all_cam_images.append(root[f"/observations/images/{cam_name}"][()])
                all_cam_images = np.stack(all_cam_images, axis=0)  # [n, T, H, W, C]

            episode = {
                "qpos": qpos[step],
                # here we assume only one camera in sim
                "images": all_cam_images.squeeze()[step],  # [T, H, W, C]
                "last_keypose": qpos[last_keypose_idx],
                "next_keypose": qpos[next_keypose_idx],
            }
            assert set(episode.keys()) == set(data.keys())
            for key, value in episode.items():
                data[key].append(value)

        for key, value in data.items():
            data[key] = np.concatenate(value, axis=0)

        return data

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.train_mask = ~self.train_mask
        val_set.train_indices = np.where(val_set.train_mask)[0]
        return val_set

    def get_normalizer(self, mode="limits", **kwargs):
        data = {
            "next_keypose": self.data["next_keypose"],
            "qpos": self.data["qpos"],
            "last_keypose": self.data["last_keypose"],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer["images"] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.train_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        data = self.data
        idx = self.train_indices[idx]
        obs_dict = dict()
        for key in self.rgb_keys:
            # move channel last to channel first
            # H,W,C
            # convert uint8 image to float32
            obs_dict[key] = (
                rearrange(data[key][idx], "... h w c -> ... c h w").astype(np.float32) / 255.0
            )
            # C,H,W
        for key in self.lowdim_keys:
            obs_dict[key] = data[key][idx].astype(np.float32)

        torch_data = {
            "obs": dict_apply(obs_dict, torch.from_numpy),
            "next_keypose": torch.from_numpy(data["next_keypose"][idx].astype(np.float32)),
        }
        return torch_data

def _create_empty_data():
    return {
        "images": list(),
        "qpos": list(),
        "next_keypose": list(),
        "last_keypose": list(),
    }


def main():
    proj_dir = pathlib.Path(__file__).parent.parent.parent
    task = "sim_insertion_scripted"
    dataset_dir = str((proj_dir / "data/aloha/datasets/").expanduser())
    shape_meta = {
        "obs": {
            "images": {"shape": (3, 240, 320), "type": "rgb"},
            "qpos": {"shape": (14,), "type": "low_dim"},
            "last_keypose": {"shape": (14,), "type": "low_dim"},
        }
    }
    np.set_printoptions(formatter={'int':lambda x: f"{x:>4}"})
    dataset = KeyposeAlohaDataset(
        dataset_dir=dataset_dir,
        shape_meta=shape_meta,
        num_episodes=50,
        camera_names=["top"],
        seed=42,
        val_ratio=0.1,
        task=task,
        use_cache=False,
    )

    print("dataset length:", len(dataset))
    print("train indices:", dataset.train_indices)
    print("data['images'].shape:", dataset.data["images"].shape)
    print("data['qpos'].shape:", dataset.data["qpos"].shape)
    print("data['last_keypose'].shape:", dataset.data["last_keypose"].shape)
    print("data['next_keypose'].shape:", dataset.data["next_keypose"].shape)

    # test normalizer
    normalizer = dataset.get_normalizer()
    nnext_kp = normalizer['next_keypose'].normalize(dataset.data['next_keypose'][:])
    nlast_kp = normalizer['last_keypose'].normalize(dataset.data['last_keypose'][:])

    # test getitem
    sample = dataset[0]
    print("sample['obs']['images'].shape:", sample["obs"]["images"].shape)
    print("sample['obs']['qpos'].shape:", sample["obs"]["qpos"].shape)
    print("sample['obs']['last_keypose'].shape:", sample["obs"]["last_keypose"].shape)
    print("sample['next_keypose'].shape:", sample["next_keypose"].shape)

if __name__ == "__main__":
    main()
