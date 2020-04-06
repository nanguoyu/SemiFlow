"""
@File : variable.py
@Author: Dong Wang
@Date : 2020/4/4
@Description : Implement of a variable vertex of the computational graph
"""
from . import DEFAULT_GRAPH
from . import Node


class Variable(Node):
    """This is a class for trainable variables
    """

    def __init__(self, initial_value=None):
        """variable constructor
        :param initial_value: initial value of current variable
        """
        super(self.__class__, self).__init__()
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
