"""
@File : test_feedforward.py
@Author: Dong Wang
@Date : 2020/4/6
"""
from SemiFlow.engine import *

with Graph().as_default():
    A = Variable([[1, 0], [0, -1]])
    b = Variable([1, 1])

    x = Variable([1, 1])

    y = MatMul(A, x)

    z = Add(y, b)

    with Session() as session:
        result = session.run(z)
        print(result)
