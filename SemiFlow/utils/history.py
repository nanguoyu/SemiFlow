"""
@File : history.py
@Author: Dong Wang
@Date : 2020/6/14
"""


class History(object):
    def __init__(self, metrics):
        self.history = {}
        allowed_metrics = {'train_loss', 'val_loss', 'train_accuracy', 'val_accuracy'}
        for metric in metrics:
            if metric in allowed_metrics:
                self.history[metric] = []

    def add_record(self, metric, value):
        if metric in self.history.keys():
            self.history[metric].append(value)
