from typing import Optional

from ..core.prelude import SrcInfo
from . import actor_kind
from .actor_kind import ActorKind, actor_kind_dict, cpu, cuda_sync
from . import lane_units


class LoopMode(object):
    # None if the loop mode doesn't correspond to cuda.
    # Otherwise, this is the "index" of how nested this loop type
    # should be in the cuda programming model.
    # (e.g. thread's cuda_nesting > block's cuda_nesting)
    cuda_nesting: Optional[int]

    def loop_mode_name(self):
        raise NotImplementedError()

    def new_actor_kind(self, old_actor_kind: ActorKind):
        raise NotImplementedError()

    def lane_unit(self):
        raise NotImplementedError()

    def cuda_can_nest_in(self, other):
        assert self.cuda_nesting is not None
        return other.cuda_nesting is not None and self.cuda_nesting > other.cuda_nesting

    def _unpack_positive_int(self, value, name):
        if hasattr(value, "val"):
            value = value.val
        int_value = int(value)
        if int_value != value or int_value <= 0:
            raise ValueError(f"{name} must be positive integer")
        return int_value


class Seq(LoopMode):
    cuda_nesting = None

    def __init__(self):
        self.cuda_nesting = None

    def loop_mode_name(self):
        return "seq"

    def new_actor_kind(self, old_actor_kind: ActorKind):
        # Sequential for loop does not change actor kind
        return old_actor_kind


seq = Seq()


class Par(LoopMode):
    cuda_nesting = None

    def __init__(self):
        self.cuda_nesting = None

    def loop_mode_name(self):
        return "par"

    def new_actor_kind(self, old_actor_kind: ActorKind):
        return cpu

    def lane_unit(self):
        return lane_units.cpu_threads


par = Par()


class CudaClusters(LoopMode):
    cuda_nesting = 2
    blocks: int

    def __init__(self, blocks):
        self.blocks = self._unpack_positive_int(blocks, "block count")

    def loop_mode_name(self):
        return "cuda_clusters"

    def new_actor_kind(self, old_actor_kind: ActorKind):
        return cuda_sync

    def lane_unit(self):
        return lane_units.cuda_clusters


class CudaBlocks(LoopMode):
    cuda_nesting = 3
    warps: int

    def __init__(self, warps=1):
        self.warps = self._unpack_positive_int(warps, "warp count")

    def loop_mode_name(self):
        return "cuda_blocks"

    def new_actor_kind(self, old_actor_kind: ActorKind):
        return cuda_sync

    def lane_unit(self):
        return lane_units.cuda_blocks


cuda_blocks = CudaBlocks()


class CudaWarpgroups(LoopMode):
    cuda_nesting = 4

    def __init__(self):
        pass

    def loop_mode_name(self):
        return "cuda_warpgroups"

    def new_actor_kind(self, old_actor_kind: ActorKind):
        return cuda_sync

    def lane_unit(self):
        return lane_units.cuda_warpgroups


cuda_warpgroups = CudaWarpgroups()


class CudaWarps(LoopMode):
    cuda_nesting = 5

    def __init__(self):
        pass

    def loop_mode_name(self):
        return "cuda_warps"

    def new_actor_kind(self, old_actor_kind: ActorKind):
        return cuda_sync

    def lane_unit(self):
        return lane_units.cuda_warps


cuda_warps = CudaWarps()


class CudaThreads(LoopMode):
    cuda_nesting = 6

    def __init__(self):
        pass

    def loop_mode_name(self):
        return "cuda_threads"

    def new_actor_kind(self, old_actor_kind: ActorKind):
        return cuda_sync

    def lane_unit(self):
        return lane_units.cuda_threads


cuda_threads = CudaThreads()


def loop_mode_for_actor_kind(actor_kind):
    class _AsyncLoopMode(LoopMode):
        def __init__(self):
            self.cuda_nesting = None

        def loop_mode_name(self):
            return actor_kind.name

        def new_actor_kind(self, old_actor_kind: ActorKind):
            return actor_kind

    return _AsyncLoopMode


def make_loop_mode_dict():
    loop_mode_dict = {
        "seq": Seq,
        "par": Par,
        "cuda_clusters": CudaClusters,
        "cuda_blocks": CudaBlocks,
        "cuda_warpgroups": CudaWarpgroups,
        "cuda_warps": CudaWarps,
        "cuda_threads": CudaThreads,
    }

    # Allow use of names of async actor kinds as loop modes
    for name, actor_kind in actor_kind_dict.items():
        assert name == actor_kind.name
        if not actor_kind.is_synthetic() and actor_kind.is_async():
            loop_mode_dict[name] = loop_mode_for_actor_kind(actor_kind)
    return loop_mode_dict


loop_mode_dict = make_loop_mode_dict()


def format_loop_cond(lo_str: str, hi_str: str, loop_mode: LoopMode):
    strings = [loop_mode.loop_mode_name(), "(", lo_str, ",", hi_str]
    for attr in loop_mode.__dict__:
        if attr != "cuda_nesting":
            strings.append(f",{attr}={getattr(loop_mode, attr)}")
    strings.append(")")
    return "".join(strings)
