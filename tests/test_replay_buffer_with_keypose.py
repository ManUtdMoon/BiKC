import sys
import pathlib
import numpy as np
import zarr

ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
sys.path.append(ROOT_DIR)

from diffusion_policy.common.replay_buffer_with_keypose import ReplayBufferWithKeypose
from diffusion_policy.common.sampler_with_keypose import SequenceSamplerWithKeypose

def test():
    # numpy
    buff = ReplayBufferWithKeypose.create_empty_numpy()
    buff.add_episode(
        {"obs": np.zeros((30,10), dtype=np.float32)},
        new_keyposes=np.array([0, 13, 29])
    )
    buff.add_episode(
        {
            "obs": np.ones((10,10)),
            "action": np.ones((10,2))
        },
        new_keyposes=np.array([0, 5, 9])
    )

    print(buff.keyposes)
    print(buff.subtraj_ends)
    print(buff.episode_ends)
    print(buff.n_episodes)
    print(buff.n_subtrajs)

    print(buff.episode_lengths)
    print(buff.subtraj_lengths)
    # buff.rechunk(256)
    data, local_keyposes = buff.pop_episode()
    print(data.keys())
    print(local_keyposes)

    print(buff.keyposes)
    print(buff.episode_ends)
    print(buff.n_episodes)
    print(buff.n_subtrajs)

    sampler = SequenceSamplerWithKeypose(
        replay_buffer=buff,
        sequence_length=10,
        pad_before=1,
        pad_after=7,
        key_first_k={"obs": 2},
        keys=["obs", "action"]
    )

    print(sampler.indices)

    print("=====================================")
    # zarr
    buff = ReplayBufferWithKeypose.create_empty_zarr()
    buff.add_episode(
        {
            "obs": np.zeros((100,10), dtype=np.float16)
        },
        new_keyposes=np.array([0, 13, 27, 50, 72, 99])
    )
    buff.add_episode(
        {
            "obs": np.ones((50,10)),
            "action": np.ones((50,2))
        },
        new_keyposes=np.array([0, 5, 19, 38, 49])
    )
    print(buff.get_chunks())
    print(buff.n_steps)
    print(buff.keyposes[:])
    print(buff.episode_ends[:])
    print(buff.n_episodes)
    print(buff.n_subtrajs)

    print(buff.episode_lengths)
    print(buff.subtraj_lengths)

    data, local_keyposes = buff.pop_episode()
    print(data.keys())
    print(local_keyposes)
    print(buff.get_chunks())

if __name__ == "__main__":
    test()