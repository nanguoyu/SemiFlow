"""
@File : __init__.py.py
@Author: Dong Wang
@Date : 2020/3/31
"""
"""core is a internal package for basic data structure"""

from .core import backend
from .graph import Graph
from .operations import Operation
import builtins

DEFAULT_GRAPH = builtins.DEFAULT_GRAPH = Graph()
