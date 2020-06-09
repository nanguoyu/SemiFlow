"""
@File : dataset.py
@Author: Dong Wang
@Date : 2020/6/9
"""
from .download import download
import sys
import gzip
import os
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


def get_one_hot(targets, nb_classes):
    return backend.eye(nb_classes)[backend.array(targets).reshape(-1)]
