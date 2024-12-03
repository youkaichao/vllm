from vllm import LLM
import pytest
from transformers import AutoModelForCausalLM
from vllm.distributed.utils import StatelessProcessGroup
from vllm.distributed.parallel_state import TensorMetadata
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

class TrainingWorker:

    def __init__(self, model: str, inference_world_size: int):
        self.model = AutoModelForCausalLM.from_pretrained(model)
        self.inference_world_size = inference_world_size

    def setup_connection(self, host: str, port: int,):
        self.control_group = StatelessProcessGroup.create(
            host=host,
            port=port,
            rank=0,
            world_size=self.inference_world_size + 1,
        )
        self.data_group = PyNcclCommunicator(self.control_group, device=torch.device(f"cuda:{self.local_rank}"))
        self.weights_metadata: List[Tuple[str, TensorMetadata]] = self.control_group.broadcast_obj(None, 0)

    def weights_iterator(self,):
        for name, metadata in self.weights_metadata:
            tensor = torch.zeros(size=metadata.size, dtype=metadata.dtype, device=torch.device(f"cuda:{self.local_rank}"))
            tensor = self.data_group.all_reduce(tensor)
            yield name, tensor

    def update_weights(self,):
        self.model_runner.model.load_weights(self.weights_iterator())


@pytest.mark.parametrize("kwargs", [
    {"tensor_parallel_size": 1,},
    {"tensor_parallel_size": 2, "distributed_executor_backend": "mp"},
    {"tensor_parallel_size": 2, "distributed_executor_backend": "ray"},
])
def test_rlhf(kwargs):
