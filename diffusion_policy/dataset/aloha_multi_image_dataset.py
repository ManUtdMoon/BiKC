if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)

import time
import os
import h5py
from typing import Dict
import torch
import numpy as np
import copy
from tqdm import tqdm
import zarr
import os
import shutil
from filelock import FileLock
from threadpoolctl import threadpool_limits
import concurrent.futures
import multiprocessing

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler,
    get_val_mask,
    downsample_mask,
)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, Jpeg2k
from diffusion_policy.common.normalize_util import get_image_range_normalizer


register_codecs()


class AlohaMultiImageDataset(BaseImageDataset):
    def __init__(
        self,
        dataset_dir: str,
        shape_meta: dict,
        num_episodes=50,
        camera_names=["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"],
        horizon=1,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.0,
        n_obs_steps=None,
        max_train_episodes=None,
        use_cache=False,
        task="aloha_insert_10s_random_init"
    ):
        super().__init__()

        ## read .hdf5 files and save to replay buffer
        replay_buffer = None
        if use_cache:
            cache_zarr_path = dataset_dir + "/" + task + ".zarr.zip"
            cache_lock_path = cache_zarr_path + ".lock"
            print("Acquiring lock on cache.")
            with FileLock(cache_lock_path):
                if not os.path.exists(cache_zarr_path):
                    # cache does not exists
                    try:
                        print("Cache does not exist. Creating!")
                        replay_buffer = _convert_to_replay(
                            num_episodes=num_episodes,
                            dataset_dir=dataset_dir,
                            camera_names=camera_names,
                            shape_meta=shape_meta,
                            store=zarr.MemoryStore(),
                        )
                        print("Saving cache to disk.")
                        with zarr.ZipStore(cache_zarr_path) as zip_store:
                            replay_buffer.save_to_store(store=zip_store)
                    except Exception as e:
                        shutil.rmtree(cache_zarr_path)
                        raise e
                else:
                    print("Loading cached ReplayBuffer from Disk.")
                    with zarr.ZipStore(cache_zarr_path, mode="r") as zip_store:
                        replay_buffer = ReplayBuffer.copy_from_store(
                            src_store=zip_store, store=zarr.MemoryStore()
                        )
                    print("Loaded!")
        else:
            replay_buffer = self.load_data(
                num_episodes=num_episodes,
                dataset_dir=dataset_dir,
                camera_names=camera_names,
            )

        ## check obs type and save to corresponding list
        rgb_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            #type = attr.get("type", "low_dim") # why "low_dim"?
            type = attr.get("type")
            if type == "rgb":
                rgb_keys.append(key)
            elif type == "low_dim":
                lowdim_keys.append(key)

        key_first_k = dict()
        if n_obs_steps is not None:
            # only take first k obs from images
            for key in rgb_keys + lowdim_keys:
                key_first_k[key] = n_obs_steps

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed,
        )
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, max_n=max_train_episodes, seed=seed
        ) ## TODO:what the purpose of downsample_mask is?

        sampler = SequenceSampler(
            replay_buffer=replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k,
        )
        
        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.n_obs_steps = n_obs_steps
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.camera_names = camera_names

    def load_data(
        self,
        num_episodes,
        dataset_dir,
        camera_names,
    ):
        replay_buffer = ReplayBuffer.create_empty_numpy()
        for i in tqdm(range(num_episodes)):  # num_episodes

            dataset_path = os.path.join(dataset_dir, f"episode_{i}.hdf5")
            with h5py.File(dataset_path, "r") as root:
                qpos = root["/observations/qpos"][()]
                qvel = root["/observations/qvel"][()]
                action = root["/action"][()]

                # stack different cameras
                all_cam_images = dict()
                for cam_name in camera_names:
                    all_cam_images[cam_name] = root[f"/observations/images/{cam_name}"][()]

            episode = {
                "qpos": qpos, # [T, dim]
                "action": action, # [T, dim]
            }
            episode.update(all_cam_images) # each cam [T, H, W, C]
            replay_buffer.add_episode(episode)

        return replay_buffer

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode="limits", **kwargs):
        data = {
            "action": self.replay_buffer["action"],
            "qpos": self.replay_buffer["qpos"],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)

        normalizer["cam_high"] = get_image_range_normalizer()
        normalizer["cam_low"] = get_image_range_normalizer()
        normalizer["cam_left_wrist"] = get_image_range_normalizer()
        normalizer["cam_right_wrist"] = get_image_range_normalizer()
        
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)

        # to save RAM, only return first n_obs_steps of OBS
        # since the rest will be discarded anyway.
        # when self.n_obs_steps is None
        # this slice does nothing (takes all)
        T_slice = slice(self.n_obs_steps)

        obs_dict = dict()
        for key in self.rgb_keys:
            # move channel last to channel first
            # T,H,W,C
            # convert uint8 image to float32
            obs_dict[key] = (
                np.moveaxis(data[key][T_slice], -1, 1).astype(np.float32) / 255.0
            )
            # T,C,H,W
            del data[key]
        for key in self.lowdim_keys:
            obs_dict[key] = data[key][T_slice].astype(np.float32)
            del data[key]

        torch_data = {
            "obs": dict_apply(obs_dict, torch.from_numpy),
            "action": torch.from_numpy(data["action"].astype(np.float32)),
        }
        return torch_data


def _convert_to_replay(
    store,
    shape_meta,
    dataset_dir,
    num_episodes,
    camera_names,
    n_workers=None,
    max_inflight_tasks=None,
):
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    if max_inflight_tasks is None:
        max_inflight_tasks = n_workers * 5

    # parse shape_meta
    rgb_keys = list()
    lowdim_keys = list()
    # construct compressors and chunks
    obs_shape_meta = shape_meta["obs"]
    for key, attr in obs_shape_meta.items():
        shape = attr["shape"]
        type = attr.get("type", "low_dim")
        if type == "rgb":
            rgb_keys.append(key)
        elif type == "low_dim":
            lowdim_keys.append(key)

    root = zarr.group(store)
    data_group = root.require_group("data", overwrite=True)
    meta_group = root.require_group("meta", overwrite=True)

    # count total steps
    episode_ends = list()
    prev_end = 0
    for i in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f"episode_{i}.hdf5")
        with h5py.File(dataset_path, "r") as demo:
            episode_length = demo["/action"].shape[0]
            episode_end = prev_end + episode_length
            prev_end = episode_end
            episode_ends.append(episode_end)
    n_steps = episode_ends[-1]
    episode_starts = [0] + episode_ends[:-1]
    _ = meta_group.array(
        "episode_ends", episode_ends, dtype=np.int64, compressor=None, overwrite=True
    )

    # save lowdim data
    for key in tqdm(lowdim_keys + ["action"], desc="Loading lowdim data"):
        data_key = "observations/" + key
        if key == "action":
            data_key = "action"
        this_data = list()
        for i in range(num_episodes):
            dataset_path = os.path.join(dataset_dir, f"episode_{i}.hdf5")
            with h5py.File(dataset_path, "r") as demo:
                this_data.append(demo[data_key][:].astype(np.float32))
        this_data = np.concatenate(this_data, axis=0)
        if key == "action":
            assert this_data.shape == (n_steps,) + tuple(shape_meta["action"]["shape"])
        else:
            assert this_data.shape == (n_steps,) + tuple(
                shape_meta["obs"][key]["shape"]
            )
        _ = data_group.array(
            name=key,
            data=this_data,
            shape=this_data.shape,
            chunks=this_data.shape,
            compressor=None,
            dtype=this_data.dtype,
        )

    def img_copy(zarr_arr, zarr_idx, hdf5_arr, hdf5_idx):
        try:
            zarr_arr[zarr_idx] = hdf5_arr[hdf5_idx]
            # make sure we can successfully decode
            _ = zarr_arr[zarr_idx]
            return True
        except Exception as e:
            return False

    with tqdm(
        total=n_steps * len(rgb_keys), desc="Loading image data", mininterval=1.0
    ) as pbar:
        # one chunk per thread, therefore no synchronization needed
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = set()

            for key in rgb_keys:

                shape = tuple(shape_meta["obs"][key]["shape"])
                c, h, w = shape

                this_compressor = Jpeg2k(level=50)
                img_arr = data_group.require_dataset(
                    name=key,
                    shape=(n_steps, h, w, c),
                    chunks=(1, h, w, c),
                    compressor=this_compressor,
                    dtype=np.uint8,
                )

                for i in range(num_episodes):
                    dataset_path = os.path.join(dataset_dir, f"episode_{i}.hdf5")

                    with h5py.File(dataset_path, "r") as demo:
                        data_key = "observations/images/" + key
                        hdf5_arr = demo[data_key][:] # [T, H * n_cam, C, W]
                        for hdf5_idx in range(hdf5_arr.shape[0]):
                            if len(futures) >= max_inflight_tasks:
                                # limit number of inflight tasks
                                completed, futures = concurrent.futures.wait(
                                    futures,
                                    return_when=concurrent.futures.FIRST_COMPLETED,
                                )
                                for f in completed:
                                    if not f.result():
                                        raise RuntimeError("Failed to encode image!")
                                pbar.update(len(completed))

                            zarr_idx = episode_starts[i] + hdf5_idx  ## index referring to all episodes
                            futures.add(
                                executor.submit(
                                    img_copy, img_arr, zarr_idx, hdf5_arr, hdf5_idx
                                )
                            )

            completed, futures = concurrent.futures.wait(futures)
            for f in completed:
                if not f.result():
                    raise RuntimeError("Failed to encode image!")
            pbar.update(len(completed))

    replay_buffer = ReplayBuffer(root)
    return replay_buffer


def main():
    proj_dir = pathlib.Path(__file__).parent.parent.parent
    task = "aloha_insert_10s_random_init"
    dataset_dir = proj_dir / "data/aloha/datasets" / task
    shape_meta = {
        "obs": {
            "cam_high": {
                "shape": (3, 120, 160),
                "type": "rgb",
            },
            "cam_low": {
                "shape": (3, 120, 160),
                "type": "rgb",
            },
            "cam_left_wrist": {
                "shape": (3, 120, 160),
                "type": "rgb",
            },
            "cam_right_wrist": {
                "shape": (3, 120, 160),
                "type": "rgb",
            },
            "qpos": {
                "shape": (14,),
                "type": "low_dim",
            },
        },
        "action": {
            "shape": (14,),
            "type": "low_dim",
        },
    }

    dataset = AlohaMultiImageDataset(
        str(pathlib.Path(dataset_dir).expanduser()),
        shape_meta,
        horizon=16,
        pad_before=1,
        pad_after=7,
        camera_names=["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"],
        use_cache=False,
        val_ratio=0.2,
        task=task,
        num_episodes=2
    )

    val_set = dataset.get_validation_dataset()
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        num_workers=8,
        shuffle=True,
        pin_memory=False,
        persistent_workers=True,
    )

    np.set_printoptions(precision=3)
    num_epochs = 1
    num_steps = 10
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}:")
        train_time_per_batch = []
        start = time.time()
        for i, batch in enumerate(tqdm(train_loader)):
            time_get = time.time()
            train_time_per_batch.append(time_get - start)
            start = time_get
            if i + 1 == num_steps:
                break
        train = np.array(train_time_per_batch)
        print(f"Train mean: {train.mean():.3f}, std: {train.std():.3f}, max: {train.max():.3f}")
        print("train:", train[:10])


if __name__ == "__main__":
    main()
