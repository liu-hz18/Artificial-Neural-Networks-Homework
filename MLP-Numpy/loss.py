from __future__ import division
import numpy as np


class EuclideanLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        '''Your codes here'''
        return 0.5 * ((target - input) ** 2).mean(axis=0).sum()

    def backward(self, input, target):
        '''Your codes here'''
        return (input - target) / len(input)
