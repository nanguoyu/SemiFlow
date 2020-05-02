"""
@File : data_utils.py
@Author: Dong Wang
@Date : 2020/5/2
"""


def DataShuffle(x, y):
    """Shuffle data"""
    # TODO utils.data_utils.DataShuffle
    return x, y


class BatchSpliter(object):
    def __init__(self, x, y, batch_size):
        self.x = x
        self.y = y
        self.batch_size = batch_size

    def get_batch(self):
        # TODO A dataGenerator : get a batch of size
        n = 0
        while n < self.batch_size:
            yield None, None
            n = n + 1
        return None, None


def split_train_val(x, y, validation_split):
    # TODO validation_split
    x_train = x
    y_train = y

    x_val = x
    y_val = y
    return x_train, y_train, x_val, y_val
