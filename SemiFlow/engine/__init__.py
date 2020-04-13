"""
@File : __init__.py.py
@Author: Dong Wang
@Date : 2020/3/31
"""
"""core is a internal package for basic data structure"""

from .core import backend
from .graph import Graph
import builtins

DEFAULT_GRAPH = builtins.DEFAULT_GRAPH = Graph()

from .node import Node
from .operations import Operation, Add, MatMul, Multiply, Square, Log, Negative
from .placeholder import Placeholder
from .variable import Variable
from .session import Session
from .utils import compute_gradients
