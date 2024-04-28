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


def broadcast_object(obj: Any,
                          src: int = 0,
                          group: Optional[ProcessGroup] = None) -> Any:
    """Broadcast the input object if the pickled object size is less than 1024 bytes."""
    group = group or get_cpu_world_group()
    ranks = torch.distributed.get_process_group_ranks(group)
    assert src in ranks, f"Invalid src rank ({src})"

    MAX_BYTES_AFTER_PICKLE = 1024

    # Bypass the function if we are using only 1 GPU.
    world_size = torch.distributed.get_world_size(group=group)
    if world_size == 1:
        return obj
    rank = torch.distributed.get_rank()
    buffer = bytearray(MAX_BYTES_AFTER_PICKLE)
    if rank == src:
        buffer = pickle.dumps(obj)
        assert len(buffer) < MAX_BYTES_AFTER_PICKLE, f"Object size after pickle {len(buffer)} is too large for broadcast"
        # pad to MAX_BYTES_AFTER_PICKLE bytes
        # pickle format itself knows when to stop, so we can just pad with spaces
        buffer = buffer + b' ' * (MAX_BYTES_AFTER_PICKLE - len(buffer) % MAX_BYTES_AFTER_PICKLE)
        buffer = bytearray(buffer)
        data = torch.frombuffer(memoryview(buffer), dtype=torch.uint8)
    else:
        data = torch.frombuffer(memoryview(buffer), dtype=torch.uint8)
    # Broadcast.
    torch.distributed.broadcast(data, src=src, group=group)
    if rank == src:
        return obj
    else:
        return pickle.loads(memoryview(buffer).tobytes())


TensorMetadata = namedtuple("TensorMetadata", ["dtype", "size", "start_indx", "end_indx"])


def broadcast_tensor_dict(
    tensor_dict: Optional[Dict[Any, Union[torch.Tensor, Any]]] = None,
    src: int = 0,
    group: Optional[ProcessGroup] = None,
) -> Optional[Dict[Any, Union[torch.Tensor, Any]]]:
    """Broadcast the input tensor dictionary."""
    group = group or torch.distributed.group.WORLD
    cpu_group = group or get_cpu_world_group()
    ranks = torch.distributed.get_process_group_ranks(group)
    assert src in ranks, f"Invalid src rank ({src})"

    # Bypass the function if we are using only 1 GPU.
    world_size = torch.distributed.get_world_size(group=group)
    if world_size == 1:
        return tensor_dict

    rank = torch.distributed.get_rank()
    if rank == src:
        metadata_list = {}
        # index in bytes
        start_indx = 0
        end_indx = 0
        buffer_views = []
        assert isinstance(
            tensor_dict,
            dict), (f"Expecting a dictionary, got {type(tensor_dict)}")
        for key, value in tensor_dict.items():
            if isinstance(value, torch.Tensor):
                assert value.is_cuda, (
                    f"Tensor {key}: {value} is not on cuda. Currently we only "
                    f"support broadcasting tensors on cuda.")
                end_indx = start_indx + value.nelement() * value.element_size()
                metadata_list[key] =TensorMetadata(value.dtype, value.size(), start_indx, end_indx)
                buffer_views.append((value.view(-1).view(dtype=torch.uint8), start_indx, end_indx))
                start_indx = end_indx
                if end_indx % 8 != 0:
                    # align start to 8 bytes
                    start_indx += 8 - end_indx % 8
            else:
                metadata_list[key] = value
        total_buffer = torch.empty(start_indx, dtype=torch.uint8, device="cuda")
        if buffer_views:
            # launch copy kernel first, then broadcast metadata, then broadcast data
            # hoping the copy kernel can overlap with metadata broadcast
            for view, start_indx, end_indx in buffer_views:
                total_buffer[start_indx:end_indx].copy_(view)
        torch.distributed.broadcast_object_list([metadata_list],
                                                src=src,
                                                group=cpu_group)
        if buffer_views:
            torch.distributed.broadcast(total_buffer, src=src, group=group)
            del total_buffer
    else:
        recv_metadata_list = [None]
        torch.distributed.broadcast_object_list(recv_metadata_list,
                                                src=src,
                                                group=cpu_group)
        assert recv_metadata_list[0] is not None
        total_buffer_size = max([value.end_indx for key, value in recv_metadata_list[0].items() if isinstance(value, TensorMetadata)], default=0)
        if total_buffer_size > 0:
            total_buffer = torch.empty(total_buffer_size, dtype=torch.uint8, device="cuda")
            torch.distributed.broadcast(total_buffer, src=src, group=group)
        tensor_dict = {}
        for key, value in recv_metadata_list[0].items():
            if isinstance(value, TensorMetadata):
                tensor = total_buffer[value.start_indx:value.end_indx].view(dtype=value.dtype).view(value.size)
                tensor_dict[key] = tensor
            else:
                tensor_dict[key] = value
    return tensor_dict
