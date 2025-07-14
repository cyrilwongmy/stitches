from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z):
        self.axes = (1,)
        max_Z = array_api.max(Z, axis=self.axes, keepdims=True)
        lse = max_Z + array_api.log(
            array_api.sum(array_api.exp(Z - max_Z), axis=self.axes, keepdims=True)
        )
        # cache softmax for the backward pass
        self.softmax = Tensor(array_api.exp(Z - lse), dtype="float32")
        return Z - lse                    # log-softmax

    def gradient(self, out_grad, node):
        # out_grad has shape (batch, n_classes)
        # sum over classes per sample: shape (batch, 1)
        # row_sum should be explicitly reshaped to (batch, 1) and then broadcasted to (batch, n_classes)
        # the numpy implicit broadcasting will cause the dtype mismatch
        row_sum = summation(out_grad, axes=(1,)).reshape((self.softmax.shape[0], 1)).broadcast_to(self.softmax.shape)
        # gradient w.r.t. logits
        return out_grad - self.softmax * row_sum

def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        max_Z = array_api.max(Z, axis=self.axes, keepdims=True)
        ans = max_Z.squeeze() + array_api.log(array_api.sum(array_api.exp(Z - max_Z), axis=self.axes))
        return ans
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        if self.axes is None:
            self.axes = tuple(range(len(node.inputs[0].shape)))
        Z = node.inputs[0]
        reduced_shape = [1 if i in self.axes else Z.shape[i] for i in range(len(Z.shape))]
        S = node.reshape(reduced_shape).broadcast_to(Z.shape)
        gradient = exp(Z - S)
        return  out_grad.reshape(reduced_shape).broadcast_to(Z.shape) * gradient
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

