"""
@File : dataset.py
@Author: Dong Wang
@Date : 2020/6/9
"""
import array
import struct
import functools
import operator
from .download import download
import sys
import gzip
import os
import tarfile
import pickle
from ..engine import backend


def mnist_old(one_hot=False):
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


def load_mnist_yann_lecun(url, file, md5):
    DATA_TYPES = {0x08: 'B',  # unsigned byte
                  0x09: 'b',  # signed byte
                  0x0b: 'h',  # short (2 bytes)
                  0x0c: 'i',  # int (4 bytes)
                  0x0d: 'f',  # float (4 bytes)
                  0x0e: 'd'}  # double (8 bytes)

    try:
        file_name = download(url, file, md5)
    except Exception as e:
        print("Download mnist failed")
        sys.exit(1)
    with gzip.open(file_name, "rb") as f:
        header = f.read(4)
        zeros, data_type, num_dimensions = struct.unpack('>HBB', header)
        try:
            data_type = DATA_TYPES[data_type]
        except KeyError:
            raise ValueError('Unknown data type '
                             '0x%02x in IDX file' % data_type)
        dimension_sizes = struct.unpack('>' + 'I' * num_dimensions,
                                        f.read(4 * num_dimensions))
        data = array.array(data_type, f.read())
        data.byteswap()  # looks like array.array reads data as little endian
        expected_items = functools.reduce(operator.mul, dimension_sizes)
        if len(data) != expected_items:
            raise ValueError('IDX file has wrong number of items. '
                             'Expected: %d. Found: %d' % (expected_items,
                                                          len(data)))

        return backend.array(data).reshape(dimension_sizes)


def mnist(one_hot=False):
    url_mnist = "http://yann.lecun.com/exdb/mnist/"
    train_image_file = 'train-images-idx3-ubyte.gz'
    train_image_md5 = "f68b3c2dcbeaaa9fbdd348bbdeb94873"
    train_image_size = 9912422
    train_label_file = 'train-labels-idx1-ubyte.gz'
    train_label_md5 = "d53e105ee54ea40749a09fcbcd1e9432"
    train_label_size = 28881
    test_image_file = 't10k-images-idx3-ubyte.gz'
    test_image_md5 = "9fb629c4189551a2d022fa330f9573f3"
    test_image_size = 1648877
    test_label_file = 't10k-labels-idx1-ubyte.gz'
    test_label_md5 = "ec29112dd5afa0611ce80d1b7f02629c"
    test_label_size = 4542
    train_image_set = load_mnist_yann_lecun(url=url_mnist + train_image_file,
                                            file=train_image_file,
                                            md5=train_image_md5)
    train_label_set = load_mnist_yann_lecun(url=url_mnist + train_label_file,
                                            file=train_label_file,
                                            md5=train_label_md5)
    test_image_set = load_mnist_yann_lecun(url=url_mnist + test_image_file,
                                           file=test_image_file,
                                           md5=test_image_md5)
    test_label_set = load_mnist_yann_lecun(url=url_mnist + test_label_file,
                                           file=test_label_file,
                                           md5=test_label_md5)
    train_set = [train_image_set, train_label_set]
    test_set = [test_image_set, test_label_set]

    if one_hot:
        train_set = (train_set[0], get_one_hot(train_set[1], 10))
        test_set = (test_set[0], get_one_hot(test_set[1], 10))

    return train_set, test_set
