"""
@File : placeholder.py
@Author: Dong Wang
@Date : 2020/4/4
@Description : Implement of a placeholder vertex of the computational graph
"""
from . import DEFAULT_GRAPH
from . import Node
from . import Add, MatMul, Multiply, Square, Log, Negative


class Placeholder(Node):
    def __init__(self, name=None):
        super(Placeholder, self).__init__(name=name)
        self.output_value = None

        # Nodes that receive this placeholder node as input.
        self.output_nodes = []

        # Graph the placeholder node belongs to.
        self.graph = DEFAULT_GRAPH

        # Add to the currently active default graph.
        self.graph.placeholders.append(self)

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
