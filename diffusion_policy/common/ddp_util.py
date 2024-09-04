import os
import logging
import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

class NoOpContextManager:
    def __init__(self, iterable=None):
        self.iterable = iterable

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __iter__(self):
        assert self.iterable is not None
        return iter(self.iterable)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def init_distributed_mode():
    ddp_info = dict()
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        ddp_info["rank"] = int(os.environ["RANK"])
        ddp_info["world_size"] = int(os.environ["WORLD_SIZE"])
        ddp_info["gpu"] = int(os.environ["LOCAL_RANK"])
    else:
        print("Not using distributed mode")
        raise NotImplementedError

    torch.cuda.set_device(ddp_info["gpu"])
    ddp_info["dist_backend"] = "nccl"
    rank, world_size = ddp_info["rank"], ddp_info["world_size"]
    logger.info(
        f"| distributed init (rank {rank}) in a world size {world_size}"
    )
    torch.distributed.init_process_group(
        backend=ddp_info["dist_backend"],
        init_method="env://",
        world_size=ddp_info["world_size"],
        rank=ddp_info["rank"]
    )
    torch.distributed.barrier()

    return ddp_info


def reduce_across_processes(val):
    if not is_dist_avail_and_initialized():
        # nothing to sync, but we still convert to tensor for consistency with the distributed case.
        return torch.as_tensor(val)

    t = torch.as_tensor(val, device="cuda")
    dist.barrier()
    dist.all_reduce(t)
    return t