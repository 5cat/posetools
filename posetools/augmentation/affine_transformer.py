from .. import backend as be

class AffineTransformer:
	def __init__(self, image_size):
		self.transformation_pipeline = []
		self.image_size = image_size
		self.size_x = self.image_size[0]
		self.size_y = self.image_size[1]
		self.matrix = be.eye(3)


	def h_flip(self):
		self.scale(1, -1)
		self.translate(0, self.size_y)

	
	def v_flip(self):
		self.scale(-1, 1)
		self.translate(self.size_x, 0)

	def rotate(self, theta):
		pi = 3.141592653589793238
		theta = theta * (pi / 180)
		def func(matrix):
			rotate_matrix = [[ be.cos(theta) ,-be.sin(theta) , 0 ],
							 [ be.sin(theta) , be.cos(theta) , 0 ],
							 [       0       ,       0       , 1 ]]
			return be.tensordot(rotate_matrix, matrix, axes = 1)

		self.translate(-0.5, -0.5)
		self.transformation_pipeline.append(func)
		self.translate(0.5, 0.5)

	def translate(self, tx, ty):
		if -1 < tx < 1:
			tx = self.size_x * tx
		if -1 < ty < 1:
			ty = self.size_y * ty

		def func(matrix):
			translate_matrix = [[ 1 , 0 , tx ],
								[ 0 , 1 , ty ],
								[ 0 , 0 , 1  ]]
			return be.tensordot(translate_matrix, matrix, axes = 1)
		self.transformation_pipeline.append(func)

	def scale(self, sx, sy):
		if -1 < sx < 1:
			sx = self.size_x * sx
		if -1 < sy < 1:
			sy = self.size_y * sy

		def func(matrix):
			scaling_matrix = [[ sx , 0  , 0 ],
							  [ 0  , sy , 0 ],
							  [ 0  , 0  , 1 ]]
			return be.tensordot(scaling_matrix, matrix, axes = 1)

		self.transformation_pipeline.append(func)

	def shear(self, skew_x, skew_y):

		def func(matrix):
			shear_matrix = [[   1    , skew_y , 0 ],
							[ skew_y ,   1    , 0 ],
							[   0    ,   0    , 1 ]]
			return be.tensordot(shear_matrix, matrix, axes = 1)

		self.transformation_pipeline.append(func)

	def initialise_matrix(self):
		matrix = be.eye(3)
		for func in self.transformation_pipeline:
			matrix = func(matrix)
		self.matrix = matrix

	def apply_transformation_points(self, x):
		return be.tensordot(x, self.matrix, axes = (-1, -1))

	def apply_transformation_images(self, images):
		return be.image.affine_transform(images, self.matrix)

	def clear_transformation(self):
		self.transformation_pipeline = []
		self.matrix = be.eye(3)

if __name__ == '__main__':
	from urllib.request import urlopen
	import numpy as np
	import cv2
	def url_to_image(url, readFlag=cv2.IMREAD_COLOR):
		# download the image, convert it to a NumPy array, and then read
		# it into OpenCV format
		resp = urlopen(url)
		image = np.asarray(bytearray(resp.read()), dtype="uint8")
		image = cv2.imdecode(image, readFlag)[:,:,::-1]#.astype('float32')
		return image

	import matplotlib.pyplot as plt
	PA = AffineTransformer((256, 256))
	PA.rotate(30)
	PA.translate(0.3, 0.3)
	PA.scale(1.2,1.3)
	PA.initialise_matrix()
	img = url_to_image('https://pbs.twimg.com/profile_images/664169149002874880/z1fmxo00_400x400.jpg')
	img = PA.apply_transformation_images(img)
	plt.imshow(img)
	plt.show()
	#PA.shear(0.2, 0.1)
	#print(PA.apply_transformation([50, 75, 1]))	
