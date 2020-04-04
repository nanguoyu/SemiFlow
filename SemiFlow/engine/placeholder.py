"""
@File : placeholder.py
@Author: Dong Wang
@Date : 2020/4/4
@Description : Implement of a placeholder vertex of the computational graph
"""
from . import DEFAULT_GRAPH


class Placeholder(object):
    def __init__(self):
        self.output_value = None

        # Nodes that receive this placeholder node as input.
        self.output_nodes = []

        # Graph the placeholder node belongs to.
        self.graph = DEFAULT_GRAPH

        # Add to the currently active default graph.
        self.graph.placeholders.append(self)
