import numpy as np

from vllm.array import VarLenArray


def test_var_len_array():
    vla = VarLenArray(10, META_SIZE=1)
    vla.append(1)
    vla.append(2)
    vla.append(3)
    assert vla[0] == 1
    assert vla[1] == 2
    assert vla[2] == 3
    assert len(vla) == 3
    vla[0] = 4
    assert vla[0] == 4
    assert vla[1] == 2
    assert vla[2] == 3
    assert len(vla) == 3
    vla.extend(np.array([5, 6, 7]))
    assert vla[0] == 4
    assert vla[1] == 2
    assert vla[2] == 3
    assert vla[3] == 5
    assert vla[4] == 6
    assert vla[5] == 7
    assert len(vla) == 6
    vla[3] = 8
    assert vla[0] == 4
    assert vla[1] == 2
    assert vla[2] == 3
    assert vla[3] == 8
    assert vla[4] == 6
    assert vla[5] == 7
    assert len(vla) == 6
    vla[0] = 0
    assert vla[0] == 0
    assert vla[1] == 2
    assert vla[2] == 3
    assert vla[3] == 8
    assert vla[4] == 6
    assert vla[5] == 7
    assert len(vla) == 6
    vla[5] = 9
    assert vla[0] == 0
    assert vla[1] == 2
    assert vla[2] == 3
    assert vla[3] == 8
    assert vla[4] == 6
    assert vla[5] == 9
    assert len(vla) == 6

    assert vla.is_empty() is False
    assert vla.get_num_empty_slots() == 4
    assert vla.is_full() is False
    assert vla  # __bool__
    vla.set_meta_item(0, 10)
    assert vla.get_meta_item(0) == 10
    assert vla.to_array().tolist() == [0, 2, 3, 8, 6, 9]
    assert vla.concat(np.array([10, 11, 12
                                ])).tolist() == [10, 11, 12, 0, 2, 3, 8, 6, 9]
    assert vla.to_array().tolist() == [0, 2, 3, 8, 6, 9]
    assert vla[2:4].tolist() == [3, 8]
