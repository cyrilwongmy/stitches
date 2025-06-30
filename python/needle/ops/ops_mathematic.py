"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

BACKEND = "np"
import numpy as array_api

class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return array_api.power(a, b)
        ### END YOUR SOLUTION
        
    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        ga = out_grad * b * power(a, b - 1)
        gb = out_grad * log(a) * power(a, b)
        return ga, gb
        ### END YOUR SOLUTION

def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""
    """ if scalar = 3, then compute a^3 for each element in a """

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        print(f"PowerScalar.compute called with a={a}, scalar={self.scalar}")
        return array_api.power(a, self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # adjoint = (y / v_i+1)' * d'(v_i+1 / v_i)' = out_grad * (v_i+1 / v_i)'
        scalar_tensor = Tensor(array_api.array([self.scalar]))

        # recomputation
        # return scalar_tensor * out_grad * array_api.power(node.inputs[0].realize_cached_data(), self.scalar - 1)

        # reuse the compute result of power to reduce redundant computation
        return scalar_tensor * out_grad * array_api.divide(node.cached_data, node.inputs[0].cached_data)
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.true_divide(a, b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        ### BEGIN YOUR SOLUTION
        return out_grad * array_api.true_divide(1, rhs.cached_data), out_grad * array_api.true_divide(-lhs.cached_data, array_api.power(rhs.cached_data, 2))
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.true_divide(a, self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * Tensor(array_api.array([1 / self.scalar]))
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.axes is None:
            return array_api.swapaxes(a, -1, -2)
        else:
            return array_api.swapaxes(a, *self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return transpose(out_grad, self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return reshape(out_grad, node.inputs[0].realize_cached_data().shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.broadcast_to(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a_shape = node.inputs[0].realize_cached_data().shape
        out_grad_removed = summation(out_grad, axes=tuple(range(len(out_grad.shape) - len(a_shape))))
        for i, dim in enumerate(a_shape):
            if dim == 1:
                out_grad_removed = summation(out_grad_removed, axes=(i,))
        return reshape(out_grad_removed, a_shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.sum(a, self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        if self.axes is None:
            return Tensor(array_api.ones_like(node.inputs[0].cached_data)) * out_grad
        else:
            # reshape the out_grad to the same shape as the input
            shape_gy = list(node.inputs[0].realize_cached_data().shape)
            for ax in sorted(self.axes):
                shape_gy[ax] = 1
            gy_reshaped = reshape(out_grad, shape_gy)
            return broadcast_to(gy_reshaped, node.inputs[0].realize_cached_data().shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        return a @ b

    """
    MatMul supports broadcasting and handles cases where the input tensors are not 2D.
    The input and output tensors can have any number of dimensions, but the last two dimensions
    of the input tensors must be compatible for matrix multiplication.
    The output tensor will have the shape determined by the broadcasting rules applied to the input tensors.
    For example, if a has shape (m, n) and b has shape (n, p), the output will have shape (m, p).
    If a has shape (m, n, k) and b has shape (n, p), the output will have shape (m, k, p).
    If a has shape (m, n) and b has shape (n, p, q), the output will have shape (m, p, q).
    Thus, we need to handle the broadcasting of the input tensors when doing the gradient computation
    and ensure that the gradients have the correct shape.
    """
    def gradient(self, out_grad, node):
        a, b = node.inputs
        adjoint1 = out_grad @ transpose(b)
        adjoint2 = transpose(a) @ out_grad
        # If the input tensors are not 2D, we need to sum over the extra dimensions
        # This is necessary to ensure the gradients have the correct shape
        adjoint1 = summation(adjoint1, axes=tuple(range(len(adjoint1.shape) - len(a.shape))))
        adjoint2 = summation(adjoint2, axes=tuple(range(len(adjoint2.shape) - len(b.shape))))
        return adjoint1, adjoint2

def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return -out_grad
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * array_api.true_divide(1, node.inputs[0].cached_data)
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * array_api.exp(node.inputs[0].cached_data)
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * array_api.where(node.inputs[0].realize_cached_data() > 0, 1, 0)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)

