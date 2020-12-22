from abc import ABCMeta, abstractmethod
from unittest import TestCase
"""
Here are some templates (abstract base class) for the math operations
and for unittesting so it can be used by other libraires such as
numpy, tensorflow, and pytorch
"""
class BaseBackend(metaclass = ABCMeta):
	def __init__(self):
		self.backend_module = None
		self.float16 = None
		self.float32 = None
		self.float64 = None
		self.default_dtype = self.float32

	@abstractmethod
	def get_dtype(self, dtype):
		if dtype is None:
			return self.default_dtype
		else:
			return dtype

	@abstractmethod
	def array(self, a, dtype = None):
		pass

	@abstractmethod
	def ones(self, shape, dtype = None):
		pass

	@abstractmethod
	def zeros(self, shape, dtype = None):
		pass

	@abstractmethod
	def tensordot(self, a, b, axes):
		pass

	@abstractmethod
	def expand_dims(self, a, axis):
		pass

	@abstractmethod
	def concatenate(self, a, axis):
		pass

	@abstractmethod
	def inverse(self, a):
		pass

	@abstractmethod
	def transpose(self, a, axes = None):
		pass

	@abstractmethod
	def stack(self, a, axis = 0):
		pass

	@abstractmethod
	def repeat(self, a, repeats, axis = None):
		pass

	@abstractmethod
	def logical_not(self, a):
		pass

	@abstractmethod
	def logical_or(self, a, b):
		pass

	@abstractmethod
	def logical_and(self, a, b):
		pass

	@abstractmethod
	def any(self, a, axis = None):
		pass

	@abstractmethod
	def range(self, *args, dtype = None):
		pass

	@abstractmethod
	def reshape(self, a, newshape):
		pass

	@abstractmethod
	def gather(self, a, indices, axis = None):
		pass

	@abstractmethod
	def eye(self, N, M = None, dtype = None):
		pass

	@abstractmethod
	def max(self, a, axis = None, keepdims = False):
		pass

	@abstractmethod
	def min(self, a, axis = None, keepdims = False):
		pass

	@abstractmethod
	def mean(self, a, axis = None, keepdims = False):
		pass

	@abstractmethod
	def sin(self, x):
		pass

	@abstractmethod
	def cos(self, x):
		pass

	@abstractmethod
	def seed(self, x):
		pass

	@abstractmethod
	def random_uniform(self, minval, maxval = None, size = None, dtype = None):
		pass


class UnittestBaseBackend(TestCase, metaclass = ABCMeta):
	default_rtol = 1e-5
	default_atol = 1e-8 

	@abstractmethod
	def assertIsClose(self, a, b, rtol = default_rtol, atol = default_atol):
		pass

