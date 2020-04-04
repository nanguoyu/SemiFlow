"""
@File : session.py
@Author: Dong Wang
@Date : 2020/4/4
@Description : Implement of a session for computing
"""
from . import DEFAULT_GRAPH
from . import Operation


class Session(object):
    def __init__(self):
        self.graph = DEFAULT_GRAPH

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """ Free all output values in nodes.
        """
        all_nodes = (self.graph.constants + self.graph.variables +
                     self.graph.placeholders + self.graph.operations)
        for node in all_nodes:
            node.output_value = None

    def run(self):
        pass

    def _get_prerequisite(self, operation):
        postorder_nodes = []

        def postorder_traverse(opt):
            if isinstance(opt, Operation):
                for input_node in opt.input_nodes:
                    postorder_traverse(input_node)
            postorder_nodes.append(opt)

        postorder_traverse(operation)
        return postorder_nodes
