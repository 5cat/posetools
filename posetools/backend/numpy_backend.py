from .base_backend import BaseBackend, UnittestBaseBackend
import numpy as np
class NumPy(BaseBackend):
	def __init__(self):
		self.np = np
		self.float16 = np.float16
		self.float32 = np.float32
		self.float64 = np.float64
		self.default_dtype = self.float32

	def get_dtype(self, dtype):
		if dtype is None:
			return self.default_dtype
		else:
			return dtype

	def array(self, a, dtype = None):
		return np.array(a, dtype = self.get_dtype(dtype))

	def ones(self, shape, dtype = None):
		return np.ones(shape, self.get_dtype(dtype))

	def zeros(self, shape, dtype = None):
		return np.zeros(shape, self.get_dtype(dtype))

	def tensordot(self, a, b, axes):
		return np.tensordot(a, b, axes)

	def expand_dims(self, a, axis):
		return np.expand_dims(a, axis)

	def concatenate(self, a, axis):
		return np.concatenate(a, axis)

	def inverse(self, a):
		return np.linalg.inv(a)

	def transpose(self, a, axes = None):
		return np.transpose(a, axes = axes)

	def stack(self, a, axis = 0):
		return np.stack(a, axis = axis)

	def repeat(self, a, repeats, axis = None):
		return np.repeat(a, repeats, axis = axis)

	def logical_not(self, a):
		return np.logical_not(a)

	def logical_or(self, a, b):
		return np.logical_or(a, b)

	def logical_and(self, a, b):
		return np.logical_and(a, b)

	def any(self, a, axis = None):
		return np.any(a, axis = axis)

	def range(self,*args, dtype = None):
		return np.arange(*args, dtype = self.get_dtype(dtype))

	def reshape(self, a, newshape):
		return np.reshape(a, newshape)

	def gather(self, a, indices, axis = None):
		return np.take(a, indices, axis = axis)

	def eye(self, N, M = None, dtype = None):
		return np.eye(N, M, dtype = self.get_dtype(dtype))

	def max(self, a, axis = None, keepdims = False):
		return np.max(a, axis = axis, keepdims = keepdims)

	def min(self, a, axis = None, keepdims = False):
		return np.min(a, axis = axis, keepdims = keepdims)

	def mean(self, a, axis = None, keepdims = False):
		return np.mean(a, axis = axis, keepdims = keepdims)

	def sin(self, x):
		return np.sin(x)

	def cos(self, x):
		return np.cos(x)

	def seed(self, x):
		np.random.seed(x)
	
	def random_uniform(self, minval, maxval = None, size = None, dtype = None):
		return np.random.uniform(minval, maxval, size = size)


class UnittestNumPy(UnittestBaseBackend):
	default_rtol = 1e-5
	default_atol = 1e-8

	def assertIsClose(self, a, b, rtol = default_rtol, atol = default_atol):
		assert np.isclose(a, b, rtol = rtol, atol = atol).all()
