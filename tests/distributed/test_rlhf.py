from vllm import LLM
import pytest
from transformers import AutoModelForCausalLM
from vllm.distributed.utils import StatelessProcessGroup
from vllm.distributed.parallel_state import TensorMetadata
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
import os
import ray

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
        self.data_group = PyNcclCommunicator(self.control_group, device=torch.device(f"cuda:0"))
        self.weights_metadata: List[Tuple[str, TensorMetadata]] = []
        for name, tensor in self.model.named_parameters():
            metadata = TensorMetadata(device=tensor.device, dtype=tensor.dtype, size=tensor.size())
            self.weights_metadata.append((name, metadata))

        self.control_group.broadcast_obj(self.weights_metadata, 0)

    def update_weights(self,):
        named_parameters = {name: tensor for name, tensor in self.model.named_parameters()}
        for name, metadata in self.weights_metadata:
            tensor = named_parameters[name]
            self.data_group.all_reduce(tensor)


@pytest.mark.skipif(torch.cuda.device_count() < 3,
                    reason="Need at least 3 GPUs to run the test.")
@pytest.mark.parametrize("kwargs", [
    {"tensor_parallel_size": 1,},
    {"tensor_parallel_size": 2, "distributed_executor_backend": "mp"},
    {"tensor_parallel_size": 2, "distributed_executor_backend": "ray"},
])
def test_rlhf(kwargs):
    model = "meta-llama/Llama-3.2-1B"
    training_worker = TrainingWorker(model, inference_world_size=kwargs["tensor_parallel_size"])

    inference_devices = [i for i in range(1, 1 + kwargs["tensor_parallel_size"])]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, inference_devices))
    ray.init(ignore_reinit_error=True)

    inference_engine = ray.remote(LLM).options(num_gpus=kwargs["tensor_parallel_size"]).remote(model=model, **kwargs, worker_cls="vllm_test_utils.rlhf_worker.RLHFWorker")

    handle = inference_engine.collective_rpc.remote("setup_connection", host="127.0.0.1", port=6379)
    training_worker.setup_connection(host="127.0.0.1", port=6379)
    ray.get(handle)

