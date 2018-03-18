import numpy
import six

import chainer
from chainer.backends import cuda
from chainer import function
from chainer.functions.activation import log_softmax
from chainer.utils import type_check
from chainer import variable


def _softmax(x, xp):
    e_x = xp.exp(x - xp.max(x))
    return e_x / e_x.sum(axis=1).reshape(-1, 1)


class LargeMarginLoss(function.Function):
    normalize = True
    y = None

    def __init__(self, ignore_label=-1):
        self.ignore_label = ignore_label

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        x, t = inputs

        log_y = log_softmax._log_softmax(x)
        self.y = xp.exp(log_y)

        count = (t != self.ignore_label).sum()
        self._coeff = 1.0 / max(count, 1)

        return xp.array([1.], dtype=xp.float32),

    def backward(self, inputs, grad_outputs):
        xp = cuda.get_array_module(*inputs)
        x, t = inputs
        gloss = grad_outputs[0]

        gx = self.y.copy()
        gx *= (t != self.ignore_label).reshape((len(t), 1))
        gx *= gloss * self._coeff
        return gx, None


def large_margin_loss(x, t, ignore_label=-1):
    return LargeMarginLoss(ignore_label)(x, t)
