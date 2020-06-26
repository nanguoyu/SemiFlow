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
from .utils import BatchSpliter, History
import six


class Optimizer(object):

    def __init__(self, loss, learning_rate, metrics, **kwargs):
        self.learning_rate = learning_rate
        self.loss = losses.get(loss)
        self._history = History(metrics)
        super(Optimizer, self).__init__(**kwargs)

    def ForwardPropagation(self, **kwargs):
        raise NotImplementedError

    def BackwardPropagation(self, **kwargs):
        raise NotImplementedError

    def _UpdateParameters(self, **kwargs):
        raise NotImplementedError

    def GetHistory(self):
        """Record of train_loss and other metrics
        Returns: History instance

        """
        return self._history


class StochasticGradientDescentOptimizer(Optimizer):
    """mini-batch gradient descent
    """

    def __init__(self, loss, learning_rate, metrics, **kwargs):
        self.spliter = None
        self.epochs = None
        self.batch_size = None
        self.last_layer = None
        self.first_layer = None
        super(StochasticGradientDescentOptimizer, self).__init__(loss, learning_rate, metrics, **kwargs)

    def build(self, x_train, y_train, epochs, batch_size, first_layer, last_layer):
        assert isinstance(first_layer, Layer)
        assert isinstance(last_layer, Layer)
        # Called at the beginning of training
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
        # assert x_train.shape[-1] == self.first_layer.shape[0], "wrong input size"
        # assert y_train.shape[-1] == self.last_layer.shape[-1], "wrong output size"

    def ForwardPropagation(self, x_val, y_val):
        postorder_nodes = get_prerequisite(last_layer=self.loss)
        for j in range(self.epochs):
            print("[epoch", j, "]", end="")
            i = 0
            train_loss = []
            self.spliter.shuffle()
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
                train_loss.append(self.loss.output_value)
                # print("[epoch", j, "]", "\t Batch ", i, "train_loss", self.loss.output_value)
                self._UpdateParameters()
                i += 1
            print("train_loss", backend.mean(train_loss), " ", end="")
            self._history.add_record('train_loss', backend.mean(train_loss))
            val_loss = self._validation(x_val=x_val, y_val=y_val, postorder_nodes=postorder_nodes)
            print("val_loss", val_loss)
            self._history.add_record('val_loss', val_loss)
            # Todo I am not sure that validation is performed after update parameters

    def BackwardPropagation(self):
        """Back propagation implemented in recursive method

        This method is not suggested. It's better to use compute_gradients
        """

        def bp(node: Layer, grad=None):
            grad = node.BackwardPropagation(grad=grad)
            if len(node.inbound) > 0:
                for child in node.inbound:
                    if not isinstance(child, InputLayer) and isinstance(child, Layer):
                        bp(child, grad)

        bp(self.loss)

    def _UpdateParameters(self):
        postorder_nodes = get_prerequisite(last_layer=self.loss)
        for node in postorder_nodes:
            if len(node.inbound) > 0 and node.params:
                # print("Update: ", node.name)
                if not hasattr(node, 'params'):
                    continue
                params = node.params.keys()
                for param in params:
                    node.params[param] -= self.learning_rate * node.grads[param]

    def _validation(self, x_val, y_val, postorder_nodes):
        # Validation
        if x_val is not None and y_val is not None:
            for node in postorder_nodes:
                if isinstance(node, InputLayer):
                    node.ForwardPropagation(feed=x_val)
                elif isinstance(node, losses.Loss):
                    node.ForwardPropagation(y_true=y_val)
                    return self.loss.output_value
                elif isinstance(node, Layer):
                    node.ForwardPropagation()


def get(opt, loss, learning_rate=0.005, metrics=None):
    if metrics is None:
        metrics = ['train_loss']
    if isinstance(opt, six.string_types):
        opt = opt.lower()
        if opt == 'sgd':
            return StochasticGradientDescentOptimizer(loss=loss, learning_rate=learning_rate, metrics=metrics)
        elif opt == 'rmsprop':
            # TODO Implement RMSprop
            return StochasticGradientDescentOptimizer(loss=loss, learning_rate=learning_rate, metrics=metrics)
        else:
            # TODO other Optimizer
            return StochasticGradientDescentOptimizer(loss=loss, learning_rate=learning_rate, metrics=metrics)
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
