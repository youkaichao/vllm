# a compact array implementation using numpy
# represents a variable length array in a fixed size array
# memory layout:
# 0: max_size
# 1: num_elements
# (these two are reserved)
# 2: META_SIZE
# 3: metadata (META_SIZE elements)
# ... 
# 3 + META_SIZE: data

import numpy as np
from numba import njit, types

numba_dtype = types.int64
np_dtype = np.int64

N_RESERVED = 2

@njit
def numba_initialize_array(META_SIZE, max_size):
    total_size = max_size + META_SIZE + N_RESERVED + 1
    data = np.empty((total_size,), dtype=np_dtype)
    data[0] = max_size  # max_size
    data[1] = 0  # num_elements
    data[2] = META_SIZE
    return data
numba_initialize_array = numba_initialize_array.compile((numba_dtype, numba_dtype))

@njit
def numba_is_empty(data):
    return data[1] == 0
numba_is_empty = numba_is_empty.compile((types.Array(numba_dtype, 1, 'C'),))

@njit
def numba_get_num_empty_slots(data):
    return data[0] - data[1] # max_size - num_elements
numba_get_num_empty_slots = numba_get_num_empty_slots.compile((types.Array(numba_dtype, 1, 'C'),))

@njit
def numba_is_full(data):
    return data[1] == data[0] # num_elements == max_size
numba_is_full = numba_is_full.compile((types.Array(numba_dtype, 1, 'C'),))


@njit
def numba_extend(data, elements):
    # `elements` is a full array
    max_size = data[0]
    num_elements = data[1]
    META_SIZE = data[2]
    num_to_extend = len(elements)
    assert num_to_extend <= max_size - num_elements
    start = N_RESERVED + META_SIZE + num_elements + 1
    # numba will optimize this loop
    for i in range(num_to_extend):
        data[start + i] = elements[i]
    block[1] += num_to_extend  # Update num_tokens
numba_extend = numba_extend.compile((types.Array(numba_dtype, 1, 'C'), types.Array(numba_dtype, 1, 'C')))

@njit
def numba_append(data, item):
    max_size = data[0]
    num_elements = data[1]
    META_SIZE = data[2]
    num_to_extend = 1
    assert num_to_extend <= max_size - num_elements
    start = N_RESERVED + META_SIZE + num_elements + 1
    data[start] = item
    data[1] += num_to_extend
numba_append = numba_append.compile((types.Array(numba_dtype, 1, 'C'), numba_dtype))

@njit
def numba_get_item(data, idx):
    max_size = data[0]
    num_elements = data[1]
    META_SIZE = data[2]
    if idx < 0:
        idx += num_elements
    assert 0 <= idx < num_elements
    return data[N_RESERVED + META_SIZE + 1 + idx]
numba_get_item = numba_get_item.compile((types.Array(numba_dtype, 1, 'C'), numba_dtype))

@njit
def numba_get_meta_item(data, idx):
    META_SIZE = data[2]
    assert idx < META_SIZE
    return data[N_RESERVED + 1 + idx]
numba_get_meta_item = numba_get_meta_item.compile((types.Array(numba_dtype, 1, 'C'), numba_dtype))

@njit
def numba_set_meta_item(data, idx, item):
    data[N_RESERVED + 1 + idx] = item
numba_get_meta_item = numba_get_meta_item.compile((types.Array(numba_dtype, 1, 'C'), numba_dtype, numba_dtype))


@njit
def numba_to_array(data):
    num_elements = data[1]
    META_SIZE = data[2]
    start = META_SIZE + N_RESERVED + 1
    return data[start: start + num_elements]
numba_to_array = numba_to_array.compile((types.Array(numba_dtype, 1, 'C'),))

class VarLenArray:
    def __init__(self, max_size, META_SIZE=0):
        self.META_SIZE = META_SIZE
        self.data = numba_initialize_array(META_SIZE, max_size)

    def is_empty(self):
        return numba_is_empty(self.data)

    def get_num_empty_slots(self):
        return numba_get_num_empty_slots(self.data)

    def is_full(self):
        return numba_is_full(self.data)

    def __bool__(self):
        return numba_is_empty(self.data)

    def extend(self, elements):
        numba_extend(self.data, elements)

    def append(self, item):
        numba_append(self.data, item)

    def __getitem__(self, idx):
        return numba_get_item(self.data, idx)

    def set_meta_item(self, idx, item):
        numba_set_meta_item(self.data, idx, item)

    def get_meta_item(self, idx):
        return numba_get_meta_item(self.data, idx)

    def to_array(self):
        return numba_to_array(self.data)
