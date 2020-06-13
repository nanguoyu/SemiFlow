"""
@File : optimizers.py
@Author: Dong Wang
@Date : 2020/5/1
"""
from .engine.core import backend
from queue import Queue
from . import losses
from . import activations
from .layer.core import Layer
from .layer.input import InputLayer
from .layer.core import get_prerequisite
from .utils import BatchSpliter
import six


class Optimizer(object):

    def __init__(self, loss, learning_rate, **kwargs):
        self.learning_rate = learning_rate
        self.loss = losses.get(loss)
        super(Optimizer, self).__init__(**kwargs)

    def _updateParameters(self):
        raise NotImplementedError

    def ForwardPropagation(self):
        raise NotImplementedError

    def BackwardPropagation(self):
        raise NotImplementedError


class GradientDescentOptimizer(Optimizer):

    def __init__(self, loss, learning_rate, **kwargs):
        self.spliter = None
        self.epochs = None
        self.batch_size = None
        self.last_layer = None
        self.first_layer = None
        super(GradientDescentOptimizer, self).__init__(loss, learning_rate, **kwargs)

    def build(self, x_train, y_train, epochs, batch_size, first_layer, last_layer):
        assert isinstance(first_layer, Layer)
        assert isinstance(last_layer, Layer)
        # Called at the beginning of training
        # Todo support validation
        # self.x_val = x_val
        # self.y_val = y_val
        self.spliter = BatchSpliter(x_train, y_train, batch_size=batch_size)
        self.epochs = epochs
        self.batch_size = batch_size
        self.first_layer = first_layer
        self.last_layer = last_layer
        # Speed BP
        if isinstance(self.loss, losses.CategoricalCrossentropy) and hasattr(self.last_layer, 'activation'):
            if isinstance(self.last_layer.activation, activations.Softmax):
                self.loss = losses.get('softmax_categorical_crossentropy')
                self.last_layer.activation = activations.get('linear')
        # Bind networks and loss
        self.loss.inbound.append(self.last_layer)
        last_layer.outbound.append(self.loss)
        # Check data shape
        assert x_train.shape[-1] == self.first_layer.shape[0], "wrong input size"
        assert y_train.shape[-1] == self.last_layer.shape[-1], "wrong output size"

    def _updateParameters(self):
        pass

    def ForwardPropagation(self):
        # TODO optimizer.GradientDescentOptimizer.ForwardPropagation
        postorder_nodes = get_prerequisite(last_layer=self.loss)
        for j in range(self.epochs):
            print("[epoch", j, "]")
            i = 0
            for xbatch, ybatch in self.spliter.get_batch():
                # Forward Propagation
                for node in postorder_nodes:
                    if isinstance(node, InputLayer):
                        node.ForwardPropagation(feed=xbatch)
                    elif isinstance(node, losses.Loss):
                        node.ForwardPropagation(y_true=ybatch)
                        # Back Propagation
                        compute_gradients(node)
                        # self.BackwardPropagation()
                    elif isinstance(node, Layer):
                        node.ForwardPropagation()
                print("Batch ", i, "loss_value", self.loss.output_value)
                self.UpdateParameters()
                i += 1

    def BackwardPropagation(self):
        # TODO finish this function and replace compute_gradients
        def bp(node: Layer, grad=None):
            grad = node.BackwardPropagation(grad=grad)
            if len(node.inbound) > 0:
                for child in node.inbound:
                    # print(node.name, "child", child.name)
                    if not isinstance(child, InputLayer) and isinstance(child, Layer):
                        bp(child, grad)

        bp(self.loss)

    def UpdateParameters(self):
        postorder_nodes = get_prerequisite(last_layer=self.loss)
        for node in postorder_nodes:
            if len(node.inbound) > 0 and node.params:
                # print("Update: ", node.name)
                params = node.params.keys()
                for param in params:
                    node.params[param] -= self.learning_rate * node.grads[param]


def get(opt, loss, learning_rate=0.005):
    if isinstance(opt, six.string_types):
        opt = opt.lower()
        if opt == 'sgd':
            return GradientDescentOptimizer(loss=loss, learning_rate=learning_rate)
        elif opt == 'rmsprop':
            # TODO Implement RMSprop
            return GradientDescentOptimizer(loss=loss, learning_rate=learning_rate)
        else:
            # TODO other Optimizer
            return GradientDescentOptimizer(loss=loss, learning_rate=learning_rate)
    else:
        ValueError('Could not interpret '
                   'initializer:', opt)


def compute_gradients(target_op):
    """ Backpropagation implementation computing gradient of target operation wrt
        all the other connected nodes.
    This code is forked from https://github.com/PytLab/simpleflow/blob/master/simpleflow/operations.py
    :param target_op: The target operation whose gradient wrt other nodes would
                      be computed.
    :type target_op: Any operation type.
    :return grad_table: A table containing layer objects and gradients.
    :type grad_table: dict.
    """
    # Todo: Modify simple version and correct errors
    # A dict containing a mapping between layer and gradient value of target_op wrt the layer's output.
    # NOTE: It is the gradient wrt the layer's OUTPUT NOT input.
    grad_table = {}

    # The gradient wrt target_op itself is 1.
    grad_table[target_op] = backend.ones_like(target_op.output_value)

    # Perform a breadth-first search staring from the target_op in graph.
    # Queue for layer traverasl.
    queue = Queue()
    queue.put(target_op)

    # Set for visited nodes.
    visited = set()
    visited.add(target_op)
    while not queue.empty():
        layer = queue.get()
        # Compute gradient wrt the layer's output.
        if layer != target_op:
            grads_wrt_layer_output = []
            for output_layer in layer.outbound:
                # Retrieve the gradient wrt output_layer's OUTPUT.
                grad_wrt_output_layer_output = grad_table[output_layer]

                # Compute the gradient wrt current layer's output.
                grad_wrt_layer_output = output_layer.BackwardPropagation(grad_wrt_output_layer_output)
                if len(output_layer.inbound) > 1:
                    input_layer_index = output_layer.inbound.index(layer)
                    grads_wrt_layer_output.append(grad_wrt_layer_output[input_layer_index])
                else:
                    grads_wrt_layer_output.append(grad_wrt_layer_output)

            # Sum all gradients wrt layer's output.
            tot_grad_wrt_layer_output = sum(grads_wrt_layer_output)
            grad_table[layer] = tot_grad_wrt_layer_output

        # Put adjecent nodes to queue.
        if hasattr(layer, 'inbound'):
            if len(layer.inbound) != 0:
                for input_layer in layer.inbound:
                    if input_layer not in visited:
                        visited.add(input_layer)
                        queue.put(input_layer)
    return grad_table
