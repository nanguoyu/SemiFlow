"""
@File : variable.py
@Author: Dong Wang
@Date : 2020/4/4
@Description : Implement of a variable vertex of the computational graph
"""
from . import DEFAULT_GRAPH
from . import Node
from . import Add, MatMul, Multiply, Square, Log, Negative


class Variable(Node):
    """This is a class for trainable variables
    """

    def __init__(self, initial_value=None, name=None):
        """variable constructor
        :param initial_value: initial value of current variable
        """
        super(Variable, self).__init__(name=name)
        self.initial_value = initial_value
        # nodes for recursive
        self.output_nodes = []
        # Output value of specified operation for input_nodes
        self.output_value = None

        self.graph = DEFAULT_GRAPH
        self.graph.variables.append(self)

    def compute_output(self):
        if self.output_value is None:
            self.output_value = self.initial_value

    def __add__(self, other):
        return Add(self, other)

    def __neg__(self):
        return Negative(self)

    def __sub__(self, other):
        return Add(self, Negative(other))

    def __mul__(self, other):
        return Multiply(self, other)

    def dot(self, other):
        return MatMul(self, other)
