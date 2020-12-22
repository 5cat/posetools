import tensorflow_addons as tfa
from .base_imagebackend import BaseImageBackend
from .. import backend as be

class TensorflowAddonsImageBackend(BaseImageBackend):
	def __init__(self):
		self.tfa = tfa
		self.interpolation = 'NEAREST'

	def affine_transform(self, image, matrix, size = None):
		matrix = be.reshape(be.inverse(matrix), (3*3,))[:-1]

		if len(image.shape) == 3:
			size = list(image.shape[0:2]) if size is None else size
			return tfa.image.transform(image, matrix, self.interpolation, size)

		elif len(image.shape) == 4:
			size = list(image.shape[1:3]) if size is None else size
			return tfa.image.transform(image, matrix, self.interpolation, size)

		else:

			raise Exception('cant apply affine transformation to images that are not the rank 3 or 4 arrays')

