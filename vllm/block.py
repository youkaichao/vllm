"""Token blocks."""
from typing import List

from vllm.array import VarLenArray
from vllm.utils import Device

DEFAULT_LAST_ACCESSED_TIME = -1


class LogicalTokenBlock(VarLenArray):
    """A block that stores a contiguous chunk of tokens from left to right.

    Logical blocks are used to represent the states of the corresponding
    physical blocks in the KV cache.
    """

    def __init__(
        self,
        block_number: int,
        block_size: int,
    ) -> None:
        super().__init__(max_size=block_size, META_SIZE=1)
        self.set_meta_item(0, block_number)

    @property
    def block_number(self) -> int:
        return self.get_meta_item(0)

    @block_number.setter
    def block_number(self, block_number: int) -> None:
        self.set_meta_item(0, block_number)

    def append_tokens(self, token_ids: List[int]) -> None:
        self.extend(token_ids)

    def get_token_ids(self) -> List[int]:
        return self.to_array()

    def get_last_token_id(self) -> int:
        return self[-1]


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
