"""
@File : placeholder.py
@Author: Dong Wang
@Date : 2020/4/4
@Description : Implement of a placeholder vertex of the computational graph
"""
from . import DEFAULT_GRAPH
from . import Node


class Placeholder(Node):
    def __init__(self):
        super(Placeholder, self).__init__()
        self.output_value = None

        # Nodes that receive this placeholder node as input.
        self.output_nodes = []

        # Graph the placeholder node belongs to.
        self.graph = DEFAULT_GRAPH

        # Add to the currently active default graph.
        self.graph.placeholders.append(self)
