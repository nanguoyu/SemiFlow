"""
@File : dataset.py
@Author: Dong Wang
@Date : 2020/6/9
"""
from .download import download
import sys
import gzip
import os
import tarfile
import pickle
from ..engine import backend


def mnist(one_hot=False):
    # This function is forked from https://github.com/borgwang/tinynn/blob/master/tinynn/utils/dataset.py
    url = "http://deeplearning.net/data/mnist/"
    file = "mnist.pkl.gz"
    checksum = "a02cd19f81d51c426d7ca14024243ce9"
    try:
        download(url + file, file, checksum)
    except Exception as e:
        print("Download mnist failed")
        sys.exit(1)
    # load the dataset
    with gzip.open(file, "rb") as f:
        train_set, valid_set, test_set = pickle.load(f, encoding="latin1")

    if one_hot:
        train_set = (train_set[0], get_one_hot(train_set[1], 10))
        valid_set = (valid_set[0], get_one_hot(valid_set[1], 10))
        test_set = (test_set[0], get_one_hot(test_set[1], 10))

    return train_set, valid_set, test_set


def cifar10(one_hot=False):
    url = "http://www.cs.toronto.edu/~kriz/"
    file = "cifar-10-python.tar.gz"
    checksum = "c58f30108f718f92721af3b95e74349a"
    try:
        download(url + file, file, checksum)
    except Exception as e:
        print("Download cifar10 failed")
        sys.exit(1)

    # load the dataset
    train_x, train_y = [], []
    test_x, test_y = [], []
    with open(file, "rb") as f:
        tar = tarfile.open(fileobj=f)
        for item in tar:
            obj = tar.extractfile(item)
            if not obj or item.size < 100:
                continue
            cont = pickle.load(obj, encoding="bytes")
            item_name = item.name.split("/")[-1]
            if item_name.find('_batch_') > 0:
                train_x.append(cont[b'data'])
                train_y.extend(cont[b'labels'])
            elif item_name == 'test_batch':
                test_x = cont[b'data']
                test_y = backend.asarray(cont[b'labels'])

    train_x = backend.concatenate(train_x, axis=0)
    # normalize
    means, stds = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    train_x = train_x.astype(dtype='float32')
    train_x = train_x / 255.0
    train_x = train_x.reshape((-1, 1024, 3))
    for c in range(3):
        train_x[:, :, c] = (train_x[:, :, c] - means[c]) / stds[c]
    train_x = train_x.reshape(-1, 3072)

    train_y = backend.asarray(train_y)
    train_set = (train_x, train_y)

    test_x = test_x / 255.0
    test_x = test_x.reshape((-1, 1024, 3))
    for c in range(3):
        test_x[:, :, c] = (test_x[:, :, c] - means[c]) / stds[c]
    test_x = test_x.reshape(-1, 3072)
    test_set = (test_x, test_y)

    if one_hot:
        train_set = (train_set[0], get_one_hot(train_set[1], 10))
        test_set = (test_set[0], get_one_hot(test_set[1], 10))
    return train_set, test_set


def get_one_hot(targets, nb_classes):
    return backend.eye(nb_classes)[backend.array(targets).reshape(-1)]
