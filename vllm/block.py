"""Token blocks."""
from typing import List

import numpy as np
from numba import njit, types

from vllm.utils import Device

_BLANK_TOKEN_ID = -1

DEFAULT_LAST_ACCESSED_TIME = -1

# Number of slots reserved for metadata in a logical block
LOGICAL_TOKEN_BLOCK_META_SIZE = 3


@njit
def numba_initialize_block(block_number, block_size):
    total_size = block_size + LOGICAL_TOKEN_BLOCK_META_SIZE
    block = np.full(total_size, _BLANK_TOKEN_ID, dtype=np.int32)
    block[0] = block_number  # block_number
    block[1] = block_size  # block_size
    block[2] = 0  # num_tokens
    return block
numba_initialize_block = numba_initialize_block.compile((types.int32, types.int32))

@njit
def numba_is_empty(block):
    return block[2] == 0  # num_tokens is at index 2
numba_is_empty = numba_is_empty.compile((types.Array(types.int32, 1, 'C'),))

@njit
def numba_get_num_empty_slots(block):
    return block[1] - block[2]  # block_size - num_tokens
numba_get_num_empty_slots = numba_get_num_empty_slots.compile((types.Array(types.int32, 1, 'C'),))

@njit
def numba_is_full(block):
    return block[2] == block[1]  # num_tokens == block_size
numba_is_full = numba_is_full.compile((types.Array(types.int32, 1, 'C'),))

@njit
def numba_append_tokens(block, token_ids):
    num_tokens = block[2]
    block_size = block[1]
    num_to_append = len(token_ids)
    assert num_to_append <= block_size - num_tokens
    # numba will optimize this loop
    for i in range(num_to_append):
        block[LOGICAL_TOKEN_BLOCK_META_SIZE + num_tokens + i] = token_ids[i]
    block[2] += num_to_append  # Update num_tokens
numba_append_tokens = numba_append_tokens.compile((types.Array(types.int32, 1, 'C'), types.Array(types.int32, 1, 'C')))

@njit
def numba_append_single_token(block, token_id):
    num_tokens = block[2]
    block_size = block[1]
    num_to_append = 1
    assert num_to_append <= block_size - num_tokens
    block[LOGICAL_TOKEN_BLOCK_META_SIZE + num_tokens] = token_id
    block[2] += num_to_append  # Update num_tokens
numba_append_single_token = numba_append_single_token.compile((types.Array(types.int32, 1, 'C'), types.int32))


@njit
def numba_get_token_ids(block):
    num_tokens = block[2]
    return block[LOGICAL_TOKEN_BLOCK_META_SIZE:LOGICAL_TOKEN_BLOCK_META_SIZE +
                 num_tokens]
numba_get_token_ids = numba_get_token_ids.compile((types.Array(types.int32, 1, 'C'),))

@njit
def numba_get_last_token_id(block):
    num_tokens = block[2]
    assert num_tokens > 0
    return block[LOGICAL_TOKEN_BLOCK_META_SIZE + num_tokens - 1]
numba_get_last_token_id = numba_get_last_token_id.compile((types.Array(types.int32, 1, 'C'),))

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
