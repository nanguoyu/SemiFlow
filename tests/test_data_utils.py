"""
@File : test_data_utils.py
@Author: Dong Wang
@Date : 2020/6/9
"""
from SemiFlow.utils import data_utils, BatchSpliter, split_train_val
import numpy as np


def test_DataShuffle():
    x = np.array(range(10))
    y = x ** 2
    x_train, y_train = data_utils.DataShuffle(x, y, seed=12)
    print('\n', x, y)
    print(x_train, y_train)


def test_BatchSpliter():
    x = np.array(range(13))
    y = x ** 2
    BS = BatchSpliter(x=x, y=y, batch_size=5)
    print('\n')
    print(BS.x)
    print(BS.y)
    print(BS.index)
    # assert BS.index[0] == [0, 5] and BS.index[1] == [5, 10]
    # print('\n')
    # print(BS.x)
    # print(BS.y)
    # for m, n in BS.get_batch():
    #     print(m, n)


def test_split_train_val():
    x = np.array(range(10))
    y = x ** 2
    x_train, y_train, x_val, y_val = split_train_val(x, y, validation_split=0.2)
    print('\n', x_train, y_train, x_val, y_val)
