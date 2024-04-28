"""Test the communication operators.

Run `pytest tests/distributed/test_comm_ops.py`.
"""
import os

import pytest
import ray
import torch

from vllm.distributed import (broadcast_tensor_dict,
                              tensor_model_parallel_all_gather,
                              tensor_model_parallel_all_reduce)
from vllm.test_utils import (init_test_distributed_environment,
                             multi_process_tensor_parallel)
from typing import List, Dict, Optional


@ray.remote(num_gpus=1, max_calls=1)
def all_reduce_test_worker(tensor_parallel_size: int, rank: int,
                           distributed_init_port: str):
    # it is important to delete the CUDA_VISIBLE_DEVICES environment variable
    # so that each worker can see all the GPUs
    # they will be able to set the device to the correct GPU
    del os.environ["CUDA_VISIBLE_DEVICES"]
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    init_test_distributed_environment(1, tensor_parallel_size, rank,
                                      distributed_init_port)
    num_elements = 8
    all_tensors = [
        torch.arange(num_elements, dtype=torch.float32, device="cuda") *
        (r + 1) for r in range(tensor_parallel_size)
    ]
    expected = torch.sum(torch.stack(all_tensors, dim=0), dim=0)
    t = all_tensors[rank]
    t = tensor_model_parallel_all_reduce(t)
    assert torch.allclose(t, expected)


@ray.remote(num_gpus=1, max_calls=1)
def all_gather_test_worker(tensor_parallel_size: int, rank: int,
                           distributed_init_port: str):
    # it is important to delete the CUDA_VISIBLE_DEVICES environment variable
    # so that each worker can see all the GPUs
    # they will be able to set the device to the correct GPU
    del os.environ["CUDA_VISIBLE_DEVICES"]
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    init_test_distributed_environment(1, tensor_parallel_size, rank,
                                      distributed_init_port)
    num_dimensions = 3
    tensor_size = list(range(2, num_dimensions + 2))
    total_size = 1
    for s in tensor_size:
        total_size *= s
    for all_gather_dimension in range(num_dimensions):
        all_tensors = [
            torch.arange(total_size, dtype=torch.float32,
                         device="cuda").reshape(tensor_size) * (r + 1)
            for r in range(tensor_parallel_size)
        ]
        expected = torch.cat(all_tensors, dim=all_gather_dimension)
        t = all_tensors[rank]
        t = tensor_model_parallel_all_gather(t, all_gather_dimension)
        assert torch.allclose(t, expected)

import dataclasses
from vllm.types import EfficientPickleDataclass
@dataclasses.dataclass
class TestMetaData(EfficientPickleDataclass):
    fragment: Optional[torch.Tensor] = None
    a: Optional[torch.Tensor] = None
    b: Optional[torch.Tensor] = None
    c: str = ""
    d: List[int] = dataclasses.field(default_factory=list)
    e: Dict[str, int] = dataclasses.field(default_factory=dict)

@ray.remote(num_gpus=1, max_calls=1)
def broadcast_tensor_dict_test_worker(tensor_parallel_size: int, rank: int,
                                      distributed_init_port: str):
    # it is important to delete the CUDA_VISIBLE_DEVICES environment variable
    # so that each worker can see all the GPUs
    # they will be able to set the device to the correct GPU
    del os.environ["CUDA_VISIBLE_DEVICES"]
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    init_test_distributed_environment(1, tensor_parallel_size, rank,
                                      distributed_init_port)
    test_dict = TestMetaData(
                fragment=torch.arange(3, dtype=torch.int8, device="cuda"),
        a=torch.arange(8, dtype=torch.float32, device="cuda"),
                                b=torch.arange(16, dtype=torch.int8, device="cuda"),
                                c="test",
                                d=[1, 2, 3],
                                e={"a": 1, "b": 2})
    if rank == 0:
        test_dict = broadcast_tensor_dict(test_dict, src=0)
    else:
        recv_dict = broadcast_tensor_dict(TestMetaData, src=0)
        assert torch.allclose(recv_dict.fragment, test_dict.fragment)
        assert torch.allclose(recv_dict.a, test_dict.a)
        assert torch.allclose(recv_dict.b, test_dict.b)
        assert recv_dict.c == test_dict.c
        assert recv_dict.d == test_dict.d
        assert recv_dict.e == test_dict.e


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason="Need at least 2 GPUs to run the test.")
@pytest.mark.parametrize("tensor_parallel_size", [2])
@pytest.mark.parametrize("test_target", [
    all_reduce_test_worker, all_gather_test_worker,
    broadcast_tensor_dict_test_worker
])
def test_multi_process_tensor_parallel(tensor_parallel_size, test_target):
    multi_process_tensor_parallel(tensor_parallel_size, test_target)
