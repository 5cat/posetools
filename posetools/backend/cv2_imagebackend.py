import cv2
from .base_imagebackend import BaseImageBackend
from .. import backend as be

class Cv2ImageBackend(BaseImageBackend):
	def __init__(self):
		self.cv2 = cv2
		self.interpolation = cv2.INTER_NEAREST
	def affine_transform(self, image, matrix, size = None):
		matrix = matrix[:2]

		if len(image.shape) == 3:
			size = tuple(list(image.shape[0:2])) if size is None else size
			return cv2.warpAffine(image,matrix, size, flags=self.interpolation)

		elif len(image.shape) == 4:
			size = tuple(list(image.shape[1:3])) if size is None else size
			output_images = []
			for img_i in image:
				img_i = cv2.warpAffine(img_i,matrix, size, flags=self.interpolation)
				output_images.append(img_i)
			return output_images

		else:

			raise Exception('cant apply affine transformation to images that are not the rank 3 or 4 arrays')


