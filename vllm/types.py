import dataclasses
import torch
from typing import Tuple

@dataclasses.dataclass
class TensorMetadata:
    dtype: torch.dtype
    size: Tuple[int]
    start_indx: int
    end_indx: int

class EfficientPickleDataclass:
    def getstate(self):
        # state is a tuple, sorted by field names.
        # value of default is replaced with None
        # value of tensor is replaced with metadata

        # index in bytes
        start_indx = 0
        end_indx = 0
        buffer_views = []

        fields = dataclasses.fields(self)
        fields = sorted(fields, key=lambda f: f.name)
        state = []
        for field in fields:
            value = getattr(self, field.name)
            if value == field.default:
                value = None
            elif isinstance(value, torch.Tensor):
                assert value.is_cuda, (
                    f"Tensor {key}: {value} is not on cuda. Currently we only "
                    f"support broadcasting tensors on cuda.")
                end_indx = start_indx + value.nelement() * value.element_size()
                buffer_views.append((value.view(-1).view(dtype=torch.uint8), start_indx, end_indx))
                value = TensorMetadata(value.dtype, value.size(), start_indx, end_indx)
                start_indx = end_indx
                if end_indx % 8 != 0:
                    # align start to 8 bytes
                    start_indx += 8 - end_indx % 8
            state.append(value)
        total_buffer = torch.empty(start_indx, dtype=torch.uint8, device="cuda")
        if buffer_views:
            # launch copy kernel first, then broadcast metadata, then broadcast data
            # hoping the copy kernel can overlap with metadata broadcast
            for view, start_indx, end_indx in buffer_views:
                total_buffer[start_indx:end_indx].copy_(view)
        return total_buffer, state

    def setstate(self, total_buffer, state):
        fields = dataclasses.fields(self)
        fields = sorted(fields, key=lambda f: f.name)
        for field, value in zip(fields, state):
            if value is not None:
                if isinstance(value, TensorMetadata):
                    tensor = total_buffer[value.start_indx:value.end_indx].view(dtype=value.dtype).view(value.size)
                    setattr(self, field.name, tensor)
                else:
                    setattr(self, field.name, value)
            else:
                setattr(self, field.name, field.default)
