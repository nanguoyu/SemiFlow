"""
@File : operations.py
@Author: Dong Wang
@Date : 2020/4/1
@Description : Implement of an operation vertex of the computational graph
"""
from . import DEFAULT_GRAPH
from . import backend
from . import Node


class Operation(Node):
    """This is an abstract class for operations
        each subclass should implement compute_output and compute_gradient
    """

    def __init__(self, *input_nodes, name=None):
        """ Operation constructor.
        :param input_nodes: Input nodes for this operation.
        :type input_nodes: variables,placeholders.
        """

        super(Operation, self).__init__(name=name)  # PEP 3135
        # Check the type of input_nodes
        for node in input_nodes:
            assert isinstance(node, Node), "Wrong type, input_nodes can only be Variable, Operation, Placeholder"

        # nodes for operation
        self.input_nodes = input_nodes
        # nodes for recursive
        self.output_nodes = []
        # Output value of specified operation for input_nodes
        self.output_value = None

        self.graph = DEFAULT_GRAPH

        for node in input_nodes:
            node.output_nodes.append(self)
        # Add this operation to default graph.
        self.graph.operations.append(self)

    def compute_output(self):
        raise NotImplementedError

    def compute_gradient(self):
        raise NotImplementedError

    def __add__(self, other):
        raise NotImplementedError

    def __neg__(self):
        raise NotImplementedError

    def __sub__(self, other):
        raise NotImplementedError

    def __mul__(self, other):
        raise NotImplementedError

    # def __matmul__(self, other):
    #     raise NotImplementedError

    def dot(self, other):
        raise NotImplementedError


class Add(Operation):
    """ An addition operation.
    """

    def __init__(self, x, y, name=None):
        """ Addition constructor.
        :param x: The first input node.
        :type x: Object of `Operation`, `Variable` or `Placeholder`.
        :param y: The second input node.
        :type y: Object of `Operation`, `Variable` or `Placeholder`.
        """
        super(Add, self).__init__(x, y, name=name)

    def compute_output(self):
        """ Compute and return the value of addition operation.
        """
        x, y = self.input_nodes
        self.output_value = backend.add(x.output_value, y.output_value)
        return self.output_value

    def compute_gradient(self, grad=None):
        """Compute and return the value of Add operation
        """
        if grad is None:
            grad = backend.ones_like(self.output_value)
        x, y = [node.output_value for node in self.input_nodes]
        grad_wrt_x = grad
        while backend.ndim(grad_wrt_x) > len(backend.shape(x)):
            grad_wrt_x = backend.sum(grad_wrt_x, axis=0)
        for axis, size in enumerate(backend.shape(x)):
            if size == 1:
                grad_wrt_x = backend.sum(grad_wrt_x, axis=axis, keepdims=True)

        grad_wrt_y = grad
        while backend.ndim(grad_wrt_y) > len(backend.shape(y)):
            grad_wrt_y = backend.sum(grad_wrt_y, axis=0)
        for axis, size in enumerate(backend.shape(y)):
            if size == 1:
                grad_wrt_y = backend.sum(grad_wrt_y, axis=axis, keepdims=True)
        return [grad_wrt_x, grad_wrt_y]

    def __add__(self, other):
        return Add(self, other)

    def __neg__(self):
        return Negative(self)

    def __sub__(self, other):
        return Add(self, Negative(other))

    def __mul__(self, other):
        return Multiply(self, other)

    # def __matmul__(self, other):
    #     return MatMul(self, other)
    def dot(self, other):
        return MatMul(self, other)


class Multiply(Operation):
    """ Multiplication operation.
    """

    def __init__(self, x, y, name='multiply'):
        """ Multiplication constructor.
        :param x: The first input node.
        :type x: Object of `Operation`, `Variable` or `Placeholder`.
        :param y: The second input node.
        :type y: Object of `Operation`, `Variable` or `Placeholder`.
        """
        super(Multiply, self).__init__(x, y, name=name)

    def compute_output(self):
        """ Compute and return the multiplication operation result.
        """
        x, y = self.input_nodes
        self.output_value = backend.multiply(x.output_value, y.output_value)
        return self.output_value

    def compute_gradient(self, grad=None):
        """Compute and return the value of Multiply operation
        """
        x, y = [node.output_value for node in self.input_nodes]

        if grad is None:
            grad = backend.ones_like(self.output_value)
        grad_wrt_x = grad * y
        while backend.ndim(grad_wrt_x) > len(backend.shape(x)):
            grad_wrt_x = backend.sum(grad_wrt_x, axis=0)
        for axis, size in enumerate(backend.shape(x)):
            if size == 1:
                grad_wrt_x = backend.sum(grad_wrt_x, axis=axis, keepdims=True)

        grad_wrt_y = grad * x
        while backend.ndim(grad_wrt_y) > len(backend.shape(y)):
            grad_wrt_y = backend.sum(grad_wrt_y, axis=0)
        for axis, size in enumerate(backend.shape(y)):
            if size == 1:
                grad_wrt_y = backend.sum(grad_wrt_y, axis=axis, keepdims=True)

        return [grad_wrt_x, grad_wrt_y]

    def __add__(self, other):
        return Add(self, other)

    def __neg__(self):
        return Negative(self)

    def __sub__(self, other):
        return Add(self, Negative(other))

    def __mul__(self, other):
        return Multiply(self, other)

    # def __matmul__(self, other):
    #     return MatMul(self, other)

    def dot(self, other):
        return MatMul(self, other)


class MatMul(Operation):
    """ Matrix multiplication operation.
    """

    def __init__(self, x, y, name=None):
        """ MatMul constructor.
        :param x: The first input node.
        :type x: Object of `Operation`, `Variable` or `Placeholder`.
        :param y: The second input node.
        :type y: Object of `Operation`, `Variable` or `Placeholder`.
        """
        super(MatMul, self).__init__(x, y, name=name)

    def compute_output(self):
        """ Compute and return the multiplication operation result.
        """
        x, y = self.input_nodes
        print(x.name, y.name)
        self.output_value = backend.dot(x.output_value, y.output_value)
        return self.output_value

    def compute_gradient(self, grad=None):
        """Compute and return the value of MatMul operation
        """
        if grad is None:
            grad = backend.ones_like(self.output_value)
        x, y = [node.output_value for node in self.input_nodes]

        dx = backend.dot(grad, backend.transpose(y))
        dy = backend.dot(backend.transpose(x), grad)

        return [dx, dy]

    def __add__(self, other):
        return Add(self, other)

    def __neg__(self):
        return Negative(self)

    def __sub__(self, other):
        return Add(self, Negative(other))

    def __mul__(self, other):
        return Multiply(self, other)

    # def __matmul__(self, other):
    #     return MatMul(self, other)

    def dot(self, other):
        return MatMul(self, other)


class Negative(Operation):
    def __init__(self, x, name="negative"):
        """ Negative constructor
        :param x: The input node
        :type x: Object of `Operation`, `Variable` or `Placeholder`.
        """
        super(Negative, self).__init__(x, name=name)

    def compute_output(self):
        x, = self.input_nodes
        self.output_value = -x.output_value
        return self.output_value

    def compute_gradient(self, grad=None):
        """Compute and return the value of Negative operation
        """
        if grad is None:
            grad = backend.ones_like(self.output_value)
        dx = -grad
        return dx

    def __add__(self, other):
        return Add(self, other)

    def __neg__(self):
        return Negative(self)

    def __sub__(self, other):
        return Add(self, Negative(other))

    def __mul__(self, other):
        return Multiply(self, other)

    # def __matmul__(self, other):
    #     return MatMul(self, other)

    def dot(self, other):
        return MatMul(self, other)


class Log(Operation):
    def __init__(self, x, name=None):
        """Log constructor
        :param x: The input node
        :type x: Object of `Operation`, `Variable` or `Placeholder`.
        """
        super(Log, self).__init__(x, name=name)

    def compute_output(self):
        x, = self.input_nodes
        self.output_value = backend.log(x.output_value)
        return self.output_value

    def compute_gradient(self, grad=None):
        """Compute and return the value of Log operation
        """
        x = self.input_nodes[0].output_value
        if grad is None:
            grad = backend.ones_like(self.output_value)
        if x == float('inf') or x == float('-inf'):
            return grad * float('inf')
        else:
            return grad * 1 / x

    def __add__(self, other):
        return Add(self, other)

    def __neg__(self):
        return Negative(self)

    def __sub__(self, other):
        return Add(self, Negative(other))

    def __mul__(self, other):
        return Multiply(self, other)

    # def __matmul__(self, other):
    #     return MatMul(self, other)

    def dot(self, other):
        return MatMul(self, other)


class Square(Operation):
    def __init__(self, x, name=None):
        """ Square Constructor
        :param x: The input node
        :type x: Object of `Operation`, `Variable` or `Placeholder`.
        """
        super(Square, self).__init__(x, name=name)

    def compute_output(self):
        x, = self.input_nodes
        self.output_value = backend.square(x.output_value)
        return self.output_value

    def compute_gradient(self, grad=None):
        """Compute and return the value of Square operation
        """
        input_value = self.input_nodes[0].output_value
        if grad is None:
            grad = backend.ones_like(self.output_value)
        return grad * backend.multiply(2.0, input_value)

    def __add__(self, other):
        return Add(self, other)

    def __neg__(self):
        return Negative(self)

    def __sub__(self, other):
        return Add(self, Negative(other))

    def __mul__(self, other):
        return Multiply(self, other)

    # def __matmul__(self, other):
    #     return MatMul(self, other)

    def dot(self, other):
        return MatMul(self, other)


class Exp(Operation):
    def __init__(self, x, name=None):
        super(Exp, self).__init__(x, name=name)

    def compute_output(self):
        x, = self.input_nodes
        self.output_value = backend.exp(x.output_value)
        return self.output_value

    def compute_gradient(self, grad=None):
        """Compute and return the value of Exp operation
        """
        if grad is None:
            grad = backend.ones_like(self.output_value)
        x, = self.input_nodes[0].output_value
        return backend.exp(x) * grad

    def __add__(self, other):
        return Add(self, other)

    def __neg__(self):
        return Negative(self)

    def __sub__(self, other):
        return Add(self, Negative(other))

    def __mul__(self, other):
        return Multiply(self, other)

    # def __matmul__(self, other):
    #     return MatMul(self, other)

    def dot(self, other):
        return MatMul(self, other)
