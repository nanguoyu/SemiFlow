"""
@File : graph.py
@Author: Dong Wang
@Date : 2020/4/1
@Description : Implement of basic computational graph
"""


class Graph(object):
    """ This is implemented as a ContextManager like Tensorflow style.

    example:
    with Graph().as_default():
        # These operation nodes will be added into current graph
        c = a * b
    # Then recover graph

    """

    def __init__(self):
        """ Graph constructor.
        """
        self.operations, self.variables, self.placeholders = [], [], []

    def __enter__(self):
        global DEFAULT_GRAPH
        self.old_graph = DEFAULT_GRAPH
        DEFAULT_GRAPH = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        global DEFAULT_GRAPH
        DEFAULT_GRAPH = self.old_graph

    def as_default(self):
        return self
