#!/usr/bin/python
# -*- coding: UTF-8 -*-

# Author Cleoner S. Pietralonga
# e-mail: cleonerp@gmail.com
# Apache License

import importlib.util
import warnings

import numpy as np
import tensornetwork as tn
from sympy.physics.quantum import TensorProduct
from logicqubit.utils import *

"""
Hilbert space
"""
class Hilbert():
    __number_of_qubits = 1
    __numeric = True
    __first_left = True
    __tn_backend = "numpy"
    __cuda_enabled = False

    @staticmethod
    def ket(value):  # get ket state
        result = Matrix([[Utils.onehot(i, value)] for i in range(2)], Hilbert.__numeric)
        return result

    @staticmethod
    def bra(value):  # get bra state
        result = Matrix([Utils.onehot(i, value) for i in range(2)], Hilbert.__numeric)
        return result

    @staticmethod
    def getState():  # get state of all qubits
        if Hilbert.getIsNumeric():
            state = Hilbert.kronProduct([Hilbert.ket(0) for i in range(Hilbert.getNumberOfQubits())])
        else:
            if Hilbert.isFirstLeft():
                a = sp.symbols([str(i) + "a" + str(i) + "_0" for i in range(1, Hilbert.getNumberOfQubits() + 1)])
                b = sp.symbols([str(i) + "b" + str(i) + "_1" for i in range(1, Hilbert.getNumberOfQubits() + 1)])
            else:
                a = sp.symbols([str(Hilbert.getNumberOfQubits() + 1 - i) + "a" + str(i) + "_0" for i in
                                reversed(range(1, Hilbert.getNumberOfQubits() + 1))])
                b = sp.symbols([str(Hilbert.getNumberOfQubits() + 1 - i) + "b" + str(i) + "_1" for i in
                                reversed(range(1, Hilbert.getNumberOfQubits() + 1))])
            state = Hilbert.kronProduct([Hilbert.ket(0) * a[i] + Hilbert.ket(1) * b[i] for i in range(Hilbert.getNumberOfQubits())])
        return state

    @staticmethod
    def getAdjoint(psi):  # get adjoint matrix
        result = psi.adjoint()
        return result

    @staticmethod
    def product(Operator, psi):  # performs an operation between the operator and the psi state
        result = Operator * psi
        return result

    @staticmethod
    def kronProduct(list):  # Kronecker product
        A = list[0]  # acts in qubit 1 which is the left most
        for M in list[1:]:
            A = A.kron(M)
        return A

    @staticmethod
    def setNumberOfQubits(number):
        Hilbert.__number_of_qubits = number

    @staticmethod
    def getNumberOfQubits():
        return Hilbert.__number_of_qubits

    @staticmethod
    def setNumeric(numeric):
        Hilbert.__numeric = numeric

    @staticmethod
    def getIsNumeric():
        return Hilbert.__numeric

    @staticmethod
    def setFirstLeft(value):
        Hilbert.__first_left = value

    @staticmethod
    def isFirstLeft():
        return Hilbert.__first_left

    @staticmethod
    def configureTensorBackend(backend=None, enable_cuda=False):
        resolved_backend = _TensorNetworkNumeric.resolve_backend(backend, enable_cuda)
        backend_used, cuda_enabled = _TensorNetworkNumeric.set_backend(resolved_backend, enable_cuda)
        Hilbert.__tn_backend = backend_used
        Hilbert.__cuda_enabled = cuda_enabled

    @staticmethod
    def getTensorBackend():
        return Hilbert.__tn_backend

    @staticmethod
    def isCudaEnabled():
        return Hilbert.__cuda_enabled


class _TensorNetworkNumeric:
    """Helper methods that rely on TensorNetwork operations for numeric work."""

    _DEFAULT_BACKEND = "numpy"
    _GPU_BACKENDS = (("jax", "jax"), ("pytorch", "torch"), ("tensorflow", "tensorflow"))
    _GPU_BACKEND_NAMES = tuple(name for name, _ in _GPU_BACKENDS)
    _current_backend = _DEFAULT_BACKEND
    _cuda_enabled = False

    @classmethod
    def resolve_backend(cls, requested_backend, enable_cuda):
        if requested_backend:
            return requested_backend
        if enable_cuda:
            for backend_name, module_name in cls._GPU_BACKENDS:
                if importlib.util.find_spec(module_name):
                    return backend_name
            warnings.warn("CUDA flag requested, but no GPU-capable backend was found. Falling back to numpy.",
                          RuntimeWarning)
        return cls._DEFAULT_BACKEND

    @classmethod
    def set_backend(cls, backend, enable_cuda):
        try:
            tn.set_default_backend(backend)
            cls._current_backend = backend
            cls._cuda_enabled = enable_cuda and cls._is_gpu_backend(backend)
            if enable_cuda and not cls._cuda_enabled:
                warnings.warn(f"CUDA was requested, but backend '{backend}' does not provide GPU acceleration.",
                              RuntimeWarning)
        except Exception as error:
            warnings.warn(f"TensorNetwork backend '{backend}' is unavailable ({error}). Reverting to numpy.",
                          RuntimeWarning)
            tn.set_default_backend(cls._DEFAULT_BACKEND)
            cls._current_backend = cls._DEFAULT_BACKEND
            cls._cuda_enabled = False
        return cls._current_backend, cls._cuda_enabled

    @classmethod
    def get_backend(cls):
        return cls._current_backend

    @classmethod
    def is_cuda_enabled(cls):
        return cls._cuda_enabled

    @classmethod
    def _is_gpu_backend(cls, backend):
        return backend in cls._GPU_BACKEND_NAMES

    @staticmethod
    def _to_ndarray(value):
        if isinstance(value, np.ndarray):
            return value
        return np.array(value)

    @staticmethod
    def matmul(left, right):
        left_tensor = _TensorNetworkNumeric._to_ndarray(left)
        right_tensor = _TensorNetworkNumeric._to_ndarray(right)
        if left_tensor.ndim == 0 or right_tensor.ndim == 0:
            return left_tensor * right_tensor

        left_node = tn.Node(left_tensor, name="matrix_left")
        right_node = tn.Node(right_tensor, name="matrix_right")
        if not left_node.edges or not right_node.edges:
            return left_tensor * right_tensor

        tn.connect(left_node.edges[-1], right_node.edges[0])
        result = tn.contract_between(left_node, right_node, name="matrix_product")
        return np.array(result.tensor)

    @staticmethod
    def kron(left, right):
        left_tensor = _TensorNetworkNumeric._to_ndarray(left)
        right_tensor = _TensorNetworkNumeric._to_ndarray(right)
        if left_tensor.ndim == 0 or right_tensor.ndim == 0:
            return left_tensor * right_tensor

        left_shape, right_shape = _TensorNetworkNumeric._match_shapes(left_tensor.shape, right_tensor.shape)
        left_view = left_tensor.reshape(left_shape)
        right_view = right_tensor.reshape(right_shape)

        kron_node = tn.outer_product(
            tn.Node(left_view, name="kron_left"),
            tn.Node(right_view, name="kron_right"),
            name="kron_outer"
        )
        tensor = np.array(kron_node.tensor)
        order = _TensorNetworkNumeric._interleave_order(len(left_shape))
        tensor = np.transpose(tensor, order)
        result_shape = tuple(l * r for l, r in zip(left_shape, right_shape))
        return tensor.reshape(result_shape)

    @staticmethod
    def _match_shapes(left_shape, right_shape):
        left_shape = tuple(left_shape) if left_shape else (1,)
        right_shape = tuple(right_shape) if right_shape else (1,)
        if len(left_shape) < len(right_shape):
            left_shape = (1,) * (len(right_shape) - len(left_shape)) + left_shape
        elif len(right_shape) < len(left_shape):
            right_shape = (1,) * (len(left_shape) - len(right_shape)) + right_shape
        return left_shape, right_shape

    @staticmethod
    def _interleave_order(rank):
        order = []
        for i in range(rank):
            order.extend([i, i + rank])
        return order


class Matrix:

    def __init__(self, matrix, numeric=True):
        self.__matrix = matrix
        self.__numeric = numeric
        if isinstance(matrix, list):  # if it's a list
            if self.__numeric:
                self.__matrix = np.array(matrix)  # create matrix with numpy
            else:
                self.__matrix = sp.Matrix(matrix)  # create matrix with sympy
        else:
            if isinstance(matrix, Matrix):  # if it's a Matrix class
                self.__matrix = matrix.get()
            else:
                self.__matrix = matrix

    def __add__(self, other):  # sum of the matrices
        result = self.__matrix + other.get()
        return Matrix(result, self.__numeric)

    def __sub__(self, other):  # subtraction of the matrices
        result = self.__matrix - other.get()
        return Matrix(result, self.__numeric)

    def __mul__(self, other):  # product of the matrices
        if isinstance(other, Matrix):
            other = other.get()
            if self.__numeric:
                result = _TensorNetworkNumeric.matmul(self.__matrix, other)
            else:
                result = self.__matrix * other
        else:
            result = self.__matrix * other
        return Matrix(result, self.__numeric)

    def __truediv__(self, other):
        result = self.__matrix * (1./other)
        return Matrix(result, self.__numeric)

    def __eq__(self, other):
        return self.__matrix == other.get()

    def __str__(self):
        return str(self.__matrix)

    def kron(self, other):  # Kronecker product
        if self.__numeric:
            result = _TensorNetworkNumeric.kron(self.__matrix, other.get())
        else:
            result = TensorProduct(self.__matrix, other.get())
        return Matrix(result, self.__numeric)

    def get(self):
        return self.__matrix

    def getAngles(self):  # converts state coefficients into angles
        angles = []
        if self.__numeric:
            angles = np.angle(self.__matrix)
        else:
            print("This session is symbolic!")
        return angles

    def trace(self):  # get matrix trace
        result = self.__matrix.trace()
        return Matrix(result, self.__numeric)

    def adjoint(self):  # get matrix adjoint
        if self.__numeric:
            result = self.__matrix.transpose().conj()
        else:
            result = self.__matrix.transpose().conjugate()
        return Matrix(result, self.__numeric)


# Ensure TensorNetwork starts on the default backend.
Hilbert.configureTensorBackend()
