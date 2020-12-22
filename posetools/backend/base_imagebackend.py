from abc import ABCMeta, abstractmethod
from unittest import TestCase

class BaseImageBackend(metaclass = ABCMeta):

	@abstractmethod
	def affine_transform(self, image, matrix, size = None):
		pass
	