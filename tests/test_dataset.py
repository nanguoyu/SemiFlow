"""
@File : test_dataset.py
@Author: Dong Wang
@Date : 2020/6/9
"""
from SemiFlow.engine import backend
from SemiFlow.utils import BatchSpliter
from SemiFlow.utils.dataset import mnist, cifar10


def test_mnist():
    train_set, test_set = mnist(one_hot=True)
    print(train_set[0].shape, train_set[1].shape)
    print(test_set[0].shape, test_set[1].shape)
    spliter = BatchSpliter(train_set[0], train_set[1], batch_size=128)
    i = 0
    for xbatch, ybatch in spliter.get_batch():
        print("Batch ", i, " size: ", xbatch.shape, ybatch.shape)
        print(ybatch[0])
        i = i + 1


def test_cifar10():
    cifar10()
    train_set, test_set = cifar10(one_hot=True)
    print(train_set[0].shape, train_set[1].shape)
    print(test_set[0].shape, test_set[1].shape)
    spliter = BatchSpliter(train_set[0], train_set[1], batch_size=128)
    i = 0
    for xbatch, ybatch in spliter.get_batch():
        print("Batch ", i, " size: ", xbatch.shape, ybatch.shape)
        print(backend.argmax(ybatch[0]))
        i += 1

