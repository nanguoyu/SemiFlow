"""
@File : operations.py
@Author: Dong Wang
@Date : 2020/4/1
@Description : Implement of an operation vertex of the computational graph
"""


class Operation(object):
    """This is an abstract class for operations
        each subclass should implement compute_output and compute_gradient

    """

    def __init__(self, *input_nodes):
        """ Operation constructor.

        :param input_nodes: Input nodes for this operation.
        :type input_nodes: variables,placeholders.

        """
        self.input_nodes = input_nodes
        self.output_nodes = []
        pass

    def compute_output(self):
        raise NotImplementedError

    def compute_gradient(self):
        raise NotImplementedError
