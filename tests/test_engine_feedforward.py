"""
@File : test_engine_feedforward.py
@Author: Dong Wang
@Date : 2020/4/6
"""
import pytest

from SemiFlow.engine import *


def test_basic_operations():
    with Graph().as_default():
        A = Variable([[1, 0], [0, -1]])
        b = Variable([1, 1])

        x = Variable([1, 1])

        y = MatMul(A, x)

        z = Add(y, b)

        with Session() as session:
            result = session.run(z)
            print(result)


def test_overloading_operations():
    with Graph().as_default():
        A = Variable([[1, 0], [0, -1]])
        x = Variable([1, 1])
        b = Variable([1, 1])

        y = A.dot(x)

        z = y + b
        with Session() as session:
            result = session.run(z)
            print(result)
