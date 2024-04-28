from collections import namedtuple
from typing import Any, Dict, List, Optional, Tuple, Union

import pickle
import torch
from torch.distributed import ProcessGroup

from .parallel_state import (get_tensor_model_parallel_group,
                             get_cpu_world_group,
                             get_tensor_model_parallel_rank,
                             get_tensor_model_parallel_world_size,
                             is_pynccl_enabled_for_all_reduce)


def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group.

    NOTE: This operation will be applied in-place on the input tensor if
    disable_custom_all_reduce is set to True. Otherwise, this operation may or
    may not be applied in place depending on whether custom all reduce is
    invoked for a particular tensor, which further depends on the tensor size
    and GPU topology.

    TLDR: always assume this function modifies its input, but use the return
    value as the output.
    """
    from vllm.distributed.device_communicators import pynccl_utils
    from vllm.distributed.device_communicators.custom_all_reduce import (
        custom_all_reduce)

    # Bypass the function if we are using only 1 GPU.
    if get_tensor_model_parallel_world_size() == 1:
        return input_
    out = custom_all_reduce(input_)
    if out is not None:
        return out
    if is_pynccl_enabled_for_all_reduce():
        # TODO: support multiple parallel groups.
        pynccl_utils.all_reduce(input_)
    else:
        torch.distributed.all_reduce(input_,
                                     group=get_tensor_model_parallel_group())
    return input_


def tensor_model_parallel_all_gather(input_: torch.Tensor,
                                     dim: int = -1) -> torch.Tensor:
    """All-gather the input tensor across model parallel group."""
    world_size = get_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_
    assert -input_.dim() <= dim < input_.dim(), (
        f"Invalid dim ({dim}) for input tensor with shape {input_.size()}")
    if dim < 0:
        # Convert negative dim to positive.
        dim += input_.dim()
    input_size = input_.size()
    # Allocate output tensor.
    output_tensor = torch.empty((world_size, ) + input_size,
                                dtype=input_.dtype,
                                device=input_.device)
    # All-gather.
    torch.distributed.all_gather_into_tensor(
        output_tensor, input_, group=get_tensor_model_parallel_group())
    # Reshape
    output_tensor = output_tensor.movedim(0, dim)
    output_tensor = output_tensor.reshape(input_size[:dim] +
                                          (world_size * input_size[dim], ) +
                                          input_size[dim + 1:])
    return output_tensor


def tensor_model_parallel_gather(input_: torch.Tensor,
                                 dst: int = 0,
                                 dim: int = -1) -> torch.Tensor:
    """Gather the input tensor across model parallel group.

    NOTE: We assume that the input tensor is on the same device across
    all the ranks.
    """
    world_size = get_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_
    assert -input_.dim() <= dim < input_.dim(), (
        f"Invalid dim ({dim}) for input tensor with shape {input_.size()}")
    if dim < 0:
        # Convert negative dim to positive.
        dim += input_.dim()
    # Allocate output tensor.
    if get_tensor_model_parallel_rank() == dst:
        gather_list = [torch.empty_like(input_) for _ in range(world_size)]
    else:
        gather_list = None
    # Gather.
    torch.distributed.gather(input_,
                             gather_list,
                             dst=dst,
                             group=get_tensor_model_parallel_group())
    if get_tensor_model_parallel_rank() == dst:
        output_tensor = torch.cat(gather_list, dim=dim)
    else:
        output_tensor = None
    return output_tensor


def broadcast(input_: torch.Tensor,
              src: int = 0,
              group: Optional[ProcessGroup] = None):
    """Broadcast the input tensor."""
    group = group or torch.distributed.group.WORLD
    ranks = torch.distributed.get_process_group_ranks(group)
    assert src in ranks, f"Invalid src rank ({src})"

    # Bypass the function if we are using only 1 GPU.
    world_size = torch.distributed.get_world_size(group=group)
    if world_size == 1:
        return input_
    # Broadcast.
    torch.distributed.broadcast(input_, src=src, group=group)
    return input_


def broadcast_object_list(obj_list: List[Any],
                          src: int = 0,
                          group: Optional[ProcessGroup] = None):
    """Broadcast the input object list."""
    group = group or torch.distributed.group.WORLD
    ranks = torch.distributed.get_process_group_ranks(group)
    assert src in ranks, f"Invalid src rank ({src})"

    # Bypass the function if we are using only 1 GPU.
    world_size = torch.distributed.get_world_size(group=group)
    if world_size == 1:
        return obj_list
    # Broadcast.
    torch.distributed.broadcast_object_list(obj_list, src=src, group=group)
    return obj_list

_MAX_BYTES_AFTER_PICKLE = 2048

from vllm.types import EfficientPickleDataclass, TensorMetadata

def broadcast_object(obj: Optional[Tuple[Any, ...]] = None,
                          src: int = 0,
                          group: Optional[ProcessGroup] = None,
                          pickle_max_bytes: Optional[int] = None
                          ) -> Tuple[Any, ...]:
    """Broadcast the input object if the pickled object size is less than _MAX_BYTES_AFTER_PICKLE bytes."""
    group = group or get_cpu_world_group()
    ranks = torch.distributed.get_process_group_ranks(group)
    assert src in ranks, f"Invalid src rank ({src})"

    # Bypass the function if we are using only 1 GPU.
    world_size = torch.distributed.get_world_size(group=group)
    if world_size == 1:
        return obj
    
    pickle_max_bytes = pickle_max_bytes or _MAX_BYTES_AFTER_PICKLE
    rank = torch.distributed.get_rank()
    buffer = bytearray(pickle_max_bytes)
    if rank == src:
        pickled_buffer = pickle.dumps(obj)
        assert len(pickled_buffer) < pickle_max_bytes, f"Object size after pickle {len(pickled_buffer)} is too large for broadcast"
        # pickle format itself knows when to stop, so we can just pad
        buffer[:len(pickled_buffer)] = pickled_buffer
        data = torch.frombuffer(memoryview(buffer), dtype=torch.uint8)
    else:
        data = torch.frombuffer(memoryview(buffer), dtype=torch.uint8)
    # Broadcast.
    torch.distributed.broadcast(data, src=src, group=group)
    if rank == src:
        return obj
    else:
        return pickle.loads(memoryview(buffer).tobytes())


def broadcast_tensor_dict(
    tensor_dict: Union[EfficientPickleDataclass, type] = None,
    src: int = 0,
    group: Optional[ProcessGroup] = None,
    pickle_max_bytes: Optional[int] = None,
) -> EfficientPickleDataclass:
    """Broadcast the input tensor dictionary.
    The broadcast rank holds the object, other ranks holds the class. Only the state of the object is broadcasted.
    """
    if group is None:
        group = torch.distributed.group.WORLD
        cpu_group = get_cpu_world_group()
    else:
        cpu_group = group
    ranks = torch.distributed.get_process_group_ranks(group)
    assert src in ranks, f"Invalid src rank ({src})"

    # Bypass the function if we are using only 1 GPU.
    world_size = torch.distributed.get_world_size(group=group)
    if world_size == 1:
        return tensor_dict

    rank = torch.distributed.get_rank()
    if rank == src:
        assert isinstance(tensor_dict, EfficientPickleDataclass)
        total_buffer, state = tensor_dict.getstate()
        broadcast_object(state, src=src, group=cpu_group, pickle_max_bytes=pickle_max_bytes)
        if total_buffer.numel() > 0:
            torch.distributed.broadcast(total_buffer, src=src, group=group)
            del total_buffer
    else:
        state = broadcast_object(src=src, group=cpu_group, pickle_max_bytes=pickle_max_bytes)
        total_buffer_size = max([value.end_indx for value in state if isinstance(value, TensorMetadata)], default=0)
        total_buffer = torch.empty(total_buffer_size, dtype=torch.uint8, device="cuda")
        if total_buffer_size > 0:
            torch.distributed.broadcast(total_buffer, src=src, group=group)
        assert isinstance(tensor_dict, type)
        tensor_dict = tensor_dict()
        tensor_dict.setstate(total_buffer, state)
    return tensor_dict
