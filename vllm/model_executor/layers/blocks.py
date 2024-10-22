import torch

from vllm.attention import AttentionMetadata
from vllm.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_reduce)
from vllm.distributed.utils import get_pp_indices
from vllm.model_executor.models.utils import PPMissingLayer


class TrackParallelBlock(torch.nn.Module):

    def __init__(self, make_track_fn, num_tracks: int, prefix: str = ''):
        super().__init__()
        self.num_tracks = num_tracks
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.start_track_id, self.end_track_id = get_pp_indices(
            num_hidden_layers=num_tracks,
            pp_rank=self.tp_rank,
            pp_size=self.tp_size,
        )
        # only support one track per rank, so that kv cache memory
        # management logic can be reused
        assert self.tp_size == num_tracks
        tracks = []
        for i in range(self.start_track_id):
            tracks.append(PPMissingLayer())
        for i in range(self.start_track_id, self.end_track_id):
            tracks.append(make_track_fn(prefix=prefix + f'.tracks.{i}'))
        for i in range(self.end_track_id, num_tracks):
            tracks.append(PPMissingLayer())

        self.tracks = torch.nn.ModuleList(tracks)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        output = self.tracks[self.start_track_id](positions, hidden_states,
                                                  kv_cache, attn_metadata)
        for i in range(self.start_track_id + 1, self.end_track_id):
            output += self.tracks[i](positions, hidden_states, kv_cache,
                                     attn_metadata)
        if self.tp_size > 1:
            output = tensor_model_parallel_all_reduce(output)
        return output
