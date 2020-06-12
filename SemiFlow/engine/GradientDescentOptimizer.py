"""
@File : GradientDescentOptimizer.py
@Author: Dong Wang
@Date : 2020/4/12
"""
from SemiFlow.engine import Operation
from SemiFlow.engine import compute_gradients, DEFAULT_GRAPH


class GradientDescentOptimizer(object):

    def __init__(self, learning_rate, name=None):
        self.learning_rate = learning_rate
        self.name = name

    def minimize(self, loss_opt):
        return MinimizationOperation(learning_rate=self.learning_rate, name=self.name, loss_opt=loss_opt)


class MinimizationOperation(Operation):
    def __init__(self, name=None, learning_rate=None, loss_opt=None):
        super(MinimizationOperation, self).__init__(name=name)
        # self.name = name
        self.learning_rate = learning_rate
        self.loss_opt = loss_opt

    def compute_output(self):
        grad_table = compute_gradients(self.loss_opt)
        # print("\t [loss value]:", self.loss_opt.output_value)
        for var in DEFAULT_GRAPH.variables:
            if var in grad_table:
                grad = grad_table[var]
                print("\t [grad of ", var.name, "]", grad)
                # old_value = var.output_value
                var.output_value -= self.learning_rate * grad
                # print("\t [", var.name, "] changed from", old_value, "to", var.output_value)
