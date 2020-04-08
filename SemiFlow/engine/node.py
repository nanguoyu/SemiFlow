"""
@File : node.py
@Author: Dong Wang
@Date : 2020/4/6
@Description : Implement of a abstract node of the computational graph
"""


class Node(object):
    def __init__(self, name=None):
        self.output_value = None
        self.output_nodes = []
        self.graph = None
        self.name = name
