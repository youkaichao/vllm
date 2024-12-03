from vllm.worker.worker import Worker
from vllm.distributed.utils import StatelessProcessGroup
from vllm.distributed.parallel_state import TensorMetadata
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
import torch
from typing import List, Tuple


class RLHFWorker(Worker):
    def setup_connection(self,         host: str,
        port: int,):
        self.control_group = StatelessProcessGroup.create(
            host=host,
            port=port,
            rank=self.rank + 1,
            world_size=self.world_size + 1,
        )
        self.data_group = PyNcclCommunicator(self.control_group, device=torch.device(f"cuda:{self.local_rank}"))
        self.weights_metadata: List[Tuple[str, TensorMetadata]] = self.control_group.broadcast_obj(None, 0)

    def weights_iterator(self,):
        for name, metadata in self.weights_metadata:
            tensor = torch.empty(size=metadata.size, dtype=metadata.dtype, device=torch.device(f"cuda:{self.local_rank}"))
            self.data_group.broadcast(tensor, src=0)
            yield name, tensor

    def update_weights(self,):
        self.model_runner.model.load_weights(self.weights_iterator())
