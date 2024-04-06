"""Token blocks."""
from typing import List

import numpy as np
from numba import njit, types

from vllm.utils import Device

_BLANK_TOKEN_ID = -1

DEFAULT_LAST_ACCESSED_TIME = -1

# Number of slots reserved for metadata in a logical block
LOGICAL_TOKEN_BLOCK_META_SIZE = 3


class LogicalTokenBlock:
    """A block that stores a contiguous chunk of tokens from left to right.

    Logical blocks are used to represent the states of the corresponding
    physical blocks in the KV cache.
    """

    def __init__(
        self,
        block_number: int,
        block_size: int,
    ) -> None:
        self.data = numba_initialize_block(block_number, block_size)

    def is_empty(self) -> bool:
        return numba_is_empty(self.data)

    def get_num_empty_slots(self) -> int:
        return numba_get_num_empty_slots(self.data)

    def is_full(self) -> bool:
        return numba_is_full(self.data)

    def append_tokens(self, token_ids: List[int]) -> None:
        numba_append_tokens(self.data, token_ids)

    def append_single_token(self, token_id: int) -> None:
        numba_append_single_token(self.data, token_id)

    def get_token_ids(self) -> List[int]:
        return numba_get_token_ids(self.data)

    def get_last_token_id(self) -> int:
        return numba_get_last_token_id(self.data)


class PhysicalTokenBlock:
    """Represents the state of a block in the KV cache."""

    def __init__(
        self,
        device: Device,
        block_number: int,
        block_size: int,
        block_hash: int,
        num_hashed_tokens: int,
    ) -> None:
        self.device = device
        self.block_number = block_number
        self.block_size = block_size
        self.block_hash = block_hash
        self.num_hashed_tokens = num_hashed_tokens

        self.ref_count = 0
        self.last_accessed = DEFAULT_LAST_ACCESSED_TIME

        self.computed = False

    def __repr__(self) -> str:
        return (f'PhysicalTokenBlock(device={self.device}, '
                f'block_number={self.block_number}, '
                f'num_hashed_tokens={self.num_hashed_tokens}, '
                f'ref_count={self.ref_count}, '
                f'last_accessed={self.last_accessed}, '
                f'computed={self.computed})')


# Mapping: logical block number -> physical block.
BlockTable = List[PhysicalTokenBlock]
