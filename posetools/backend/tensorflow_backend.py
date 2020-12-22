from .base_backend import BaseBackend, UnittestBaseBackend
import tensorflow as tf

class TensorFlow(BaseBackend):
	def __init__(self):
		self.tf = tf
		self.float16 = tf.float16
		self.float32 = tf.float32
		self.float64 = tf.float64
		self.default_dtype = self.float32

	def get_dtype(self, dtype):
		if dtype is None:
			return self.default_dtype
		else:
			return dtype

	def array(self, a, dtype = None):
		if type(a) != tf.Tensor:
			return tf.convert_to_tensor(a, dtype = self.get_dtype(dtype))
		else:
			return tf.cast(a, self.get_dtype(dtype))

	def ones(self, shape, dtype = None):
		return tf.ones(shape, self.get_dtype(dtype))

	def zeros(self, shape, dtype = None):
		return tf.zeros(shape, self.get_dtype(dtype))

	def tensordot(self, a, b, axes):
		return tf.tensordot(a, b, axes)

	def expand_dims(self, a, axis):
		return tf.expand_dims(a, axis)

	def concatenate(self, a, axis):
		return tf.concat(a, axis)

	def inverse(self, a):
		return tf.linalg.inv(a)

	def transpose(self, a, axes = None):
		return tf.transpose(a, perm = axes)

	def stack(self, a, axis = 0):
		return tf.stack(a, axis = axis)

	def repeat(self, a, repeats, axis = None):
		return tf.repeat(a, repeats, axis = axis)

	def logical_not(self, a):
		return tf.math.logical_not(a)

	def logical_or(self, a, b):
		return tf.math.logical_or(a, b)

	def logical_and(self, a, b):
		return tf.math.logical_and(a, b)

	def any(self, a, axis = None):
		return tf.math.reduce_any(a, axis = axis)

	def range(self, *args, dtype = None):
		return tf.range(*args, dtype = self.get_dtype(dtype))

	def reshape(self, a, newshape):
		return tf.reshape(a, newshape)

	def gather(self, a, indices, axis = None):
		return tf.gather(a, indices, axis = axis)

	def eye(self, N, M = None, dtype = None):
		return tf.eye(N, M, dtype = self.get_dtype(dtype))

	def max(self, a, axis = None, keepdims = False):
		return tf.reduce_max(a, axis = axis, keepdims = keepdims)

	def min(self, a, axis = None, keepdims = False):
		return tf.reduce_min(a, axis = axis, keepdims = keepdims)

	def mean(self, a, axis = None, keepdims = False):
		return tf.reduce_mean(a, axis = axis, keepdims = keepdims)

	def sin(self, x):
		return tf.math.sin(x)

	def cos(self, x):
		return tf.math.cos(x)

	def seed(self, x):
		tf.random.set_seed(x)
	
	def random_uniform(self, minval, maxval = None, size = None, dtype = None):
		return tf.random.uniform(size, minval, maxval, dtype = self.get_dtype(dtype))

class UnittestTensorFlow(UnittestBaseBackend):
	default_rtol = 1e-5
	default_atol = 1e-8

	def assertIsClose(self, a, b, rtol = default_rtol, atol = default_atol):
		tf.debugging.assert_near(a, b, rtol = rtol, atol = atol)