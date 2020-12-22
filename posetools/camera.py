import numpy as np
import re
from .exceptions import PointShapeError, PointRankError, PointOutsideDeviceError, CameraParameterShapeError
from . import backend as be
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# import tensorflow as tf
"""
Helpfull links for understanding camera stuff i just hated it.

Have a read if the the things down there doesnt make sense.

I know it will not make much sense either after reading it because sometimes
the camera matrix (A) is used instead of intrinsic matrix (K) such as in 
https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#projectpoints:~:text=cameraMatrix%20%E2%80%93%20Camera%20matrix
while others donates the camera matrix as M such as in [1]
and some calls it "camera calibration matrix" [4]
but focus on the elements on of the matrix rather than the naming of the matrix

Also, numpy uses row major ordering while in matlab [3] they use column major ordering so matrix shapes might be
transposed there. The size of the arrays here are labeled in i,j fashion the standard way numpy display shapes
but in wikipedia and in math in general matrices display shape by x,y indexing.

[1]: https://en.wikipedia.org/wiki/Camera_resectioning
[2]: https://en.wikipedia.org/wiki/Pinhole_camera_model
[3]: https://www.mathworks.com/help/vision/ug/camera-calibration.html
[4]: https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/EPSRC_SSAZ/node3.html
[5]: https://en.wikipedia.org/wiki/Camera_matrix
[6]: https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html
[7]: http://www.cse.psu.edu/~rtc12/CSE486/lecture12.pdf
[8]: http://www.cse.psu.edu/~rtc12/CSE486/lecture13.pdf
"""

class Camera:
	def __init__(self, extrinsic, intrinsic, size, distortion_coefficient = None, radial = 0,**dict_camera):

		#radial(1) is a distortion occurs when light rays bend more near the edges of a lens than they do at its optical center. (zero means no distortion)
		self.radial = be.array(radial)
		if self.radial.shape != ():
			raise CameraParameterShapeError('extrinsic', self.radial.shape, ())
		"""
		#extrinsic matrix (4x4) is used to convert from 3D world coordinate system to 3D camera coordinate system
		[[rotation_matrix(3x3), translation_matrix(1x3)],
		 [	    0(3x1)        ,            1           ]]
		"""
		self.extrinsic = be.array(extrinsic)
		if self.extrinsic.shape != (4, 4):
			raise CameraParameterShapeError('extrinsic', self.extrinsic.shape, (4, 4))

		# size(1x2) is the number of pixels in each axis which is the same as in the videos provided in MPI_INF_3DHP dataset
		# such that the first axis is x and the secound is y (height, width)
		self.size = be.array(size)
		if self.size.shape != (2,):
			raise CameraParameterShapeError('size', self.size.shape, (2,))

		#intrinsic matrix (3x4) is used to project from 3D camera coordinate system to 2D plane projection (2D coordinate system)
		# [[ fx ,skew, u0 , 0  ],
		#  [ 0  , fy , v0 , 0  ],
		#  [ 0  , 0  , 1  , 0  ]]
		self.intrinsic = be.array(intrinsic)
		if self.intrinsic.shape != (3, 4):
			raise CameraParameterShapeError('intrinsic', self.intrinsic.shape, (3, 4))

		#rotational matrix (3x3) taken from extrinsic matrix
		self.rotation_matrix = self.extrinsic[:3,:3]

		#translation matrix (1x3) is the position of the origin of the world coordinate system expressed in coordinates of the camera-centered coordinate system
		self.translation_matrix = self.extrinsic[:3,3]

		#camera position (1x3) expressed in world coordinate system

		self.camera_position = be.tensordot(-be.transpose(self.rotation_matrix), self.translation_matrix, axes=1)

		#focal length (1x2) (fx,fy) of the camera in pixels unit
		self.focal_length = be.stack([self.intrinsic[0, 0], self.intrinsic[1, 1]])

		#principal point (1x2) (u0,v0) it should be idealy in the center of the image represented in pixels unit
		self.principal_point = self.intrinsic[:2,2]

		#skew coefficient represents coefficient between the x and the y axis in pixels unit, and is often 0
		#in this code I'm ignoring this and i'm assuming it is zero to make things easier
		self.skew_coefficient = self.intrinsic[0,1]

		#distortion coefficient (5) [k1, k2, p1, p2, k3] are coefficient that are related to the radial distortion [6]
		if distortion_coefficient is None:
			self.distortion_coefficient = be.zeros((5,))
		else:
			self.distortion_coefficient = be.array(distortion_coefficient)

		if self.distortion_coefficient.shape!=(5,):
			raise CameraParameterShapeError('distortion_coefficient', distortion_coefficient.shape, (5,))

		if 0<(sum(abs(self.distortion_coefficient)) + abs(self.skew_coefficient)):
			raise NotImplementedError(
				"Sorry I just didnt have the time to account for distortion and skew "\
				"coefficients in this tool. consider using public libraries to correct these values "\
				"in the images instead of the points here.")

		#projection matrix (4x4) to convert from camera coordinates to normlized device coordinates
		self.projection_matrix = self.create_projection_matrix()


	#peform checks on input data
	def checks(self, points, check_points_dims=False, is_3d=False, is_2d=False, is_1d=False):
		points = be.array(points)

		if check_points_dims:
			if len(points.shape) != 2:
				raise PointRankError(len(points.shape), 2, points.shape)

		if is_3d:
			if points.shape[1] !=3 :
				raise PointShapeError(points.shape[1], 3)

		if is_2d:
			if points.shape[1] != 2:
				raise PointShapeError(points.shape[1], 2)

		if is_2d:
			if points.shape[1] != 1:
				raise PointShapeError(points.shape[1], 1)

		return points

	def cartesian_to_homogeneous(self, points):
		w = be.ones((points.shape[0],1))
		return be.concatenate((points, w), axis=1)
		
	def homogeneous_to_cartesian(self, points):
		return points[:,:-1]/be.expand_dims(points[:,-1],axis=-1)

	#@profile
	#@be.tf.function
	def operation(self, matrix, points):
		points = self.cartesian_to_homogeneous(points) # adding the dimension w (homogeneous coordinates) from (N x D) to (N x D+w)
		points = be.tensordot(points, matrix, axes=(-1,-1))
		points = self.homogeneous_to_cartesian(points) #remove the w dimension from (N x D+w) to (N x D)
		return points


	"""
	Here in this function I'm just rotating and translating the coordinates so it matches
	the camera (rotation_matrix * points + translation_matrix); which means the z axis will look 
	as if its the depth from the camera point of view.

	[Xc, Yc, Zc, Wc] = [[rotation_matrix(3x3), translation_matrix(1x3)],   * [Xw, Yw, Zw, Ww] 
					   [	    0(3x1)        ,            1           ]]
	where (Xc, Yc, Zc, Wc) is the homogeneous coordinates of the camera and (Xw, Yw, Zw, Ww) is
	the homogeneous coordinates of the world. Note that the value of Ww doesnt matter but we will use
	1 so it is easier.
	points: (Nx3) the 3D world coordinates ([[x,y,z]]*N) which will be converted to camera coordinates

	"""
	def world_to_camera(self, points):
		# Based on step 3 from [4] the extrinsic matrix should be multpied(dot) by the world coordinates
		# to get the camera coordinates; just to let you know the extrinsic matrix already contains
		# the info about rotation and the translation so it is just a straight forward dot operation
		# but to make things work we will use homogeneous coordinates (adding the dimension w) then
		# convert back to cartesian coordinates

		points = self.checks(points, check_points_dims=True, is_3d=True)
		points = self.operation(matrix = self.extrinsic, points = points)
		return points


	#read the comments for function camera_to_normlized_device for more details about this function. 
	def create_projection_matrix(self):

		x_factor = self.focal_length[0] / self.principal_point[0]
		y_factor = self.focal_length[1] / self.principal_point[1]

		s = be.max(self.size / self.focal_length)

		alpha = ((s**2 + 1)**0.5 + 1)/s

		beta = -0.5 * ((2*(s**2 + 1)**0.5 + 2)/s)


		projection_matrix = [[x_factor ,    0    ,    0    ,    0    ],
							 [    0    ,y_factor ,    0    ,    0    ],
							 [    0    ,    0    ,  alpha  ,  beta   ],
							 [    0    ,    0    ,    1    ,    0    ]]

		return be.array(projection_matrix)

	"""
	converting from camera coordinates to normlized device coordinates.
	which means making every axis between -1 and +1 so if there is a point that has a value
	higher than +1 or lower than -1 it means it is outside of the view. However, the z axis (depth)
	can be extended to positive infinity technically so we need to clip it, that is why this operation
	form a frustum. The way it is implemented here in the self.create_projection_matrix it will take
	which ever axis (x or y) that has the maximum length to form the frusturm. There has to be a
	z reference point to which the frustum can be formed from its center. 
	I'm going to start explaining here from the middle since there is a lot to go through before arriving
	to the wanted matrix. Here are links that might help you understand to the point that i will start with
	https://www.scratchapixel.com/lessons/3d-basic-rendering/perspective-and-orthographic-projection-matrix/opengl-perspective-projection-matrix
	http://info.ee.surrey.ac.uk/Teaching/Courses/eem.cgi/lectures_pdf/lecture3.pdf
	http://learnwebgl.brown37.net/08_projections/projections_perspective.html
	http://www.terathon.com/gdc07_lengyel.pdf
	http://www.songho.ca/opengl/gl_projectionmatrix.html
	http://kgeorge.github.io/2014/03/08/calculating-opengl-perspective-matrix-from-opencv-intrinsic-matrix
	here is the matrix that we want to create.
							[[  fx/u0  ,    0    ,    0    ,    0    ],
			[x, y, z, w]=	 [    0    ,  fy/v0  ,    0    ,    0    ], * [X, Y, Z, 1]
							 [    0    ,    0    ,  alpha  ,  beta   ],
							 [    0    ,    0    ,    1    ,    0    ]]
	where (x, y, z, w) is the normlized device homogeneous coordinates such that the cartesian coordinates are between [-1, +1]
	and (X, Y, Z, 1) is the homogeneous coordinates of the camera view, fx and fy are the focal length,
	u0 and v0 are the principal point. x/w and y/w will be between [-1, +1] but we want to figure out
	the values of alpha and beta such that z/w will be normlized between [-1, +1]. doing the matrix operation for the Z
	will yield us with z = (alpha/Z) + beta. we want alpha and beta value such that
	-1 = (alpha/near) + beta
	+1 = (alpha/far) + beta
	where near and far are the Z value limits or the two ends of the frustum. using algebra we will get
	alpha = 2*far*near / (far - near)
	beta = (far + near) / (far - near)

	but in our case we want 
	1 - to make a Z camera reference point such that it will be in the middle of the frustum
	2 - and the size of the frustum to be more equal or squarish

	1 - for it to be in the middle of the frustum we solve 0 = (alpha/Z_ref) + beta where Z_ref is the Z reference point
	this will give us Z_ref = (-2*f*n)/(f+n).

	2 - the size of the frustum should be squarish which means all sides of the frustum should be kinda equal
	we can represent this by using f-n = Z_ref * max(width/fx, height/fy). the max part is to take which
	ever axis is bigger and deviding the original image sizes (width and height) by the focal point will give us
	the pixels/millimeter since fx = Fx*(w/W) where Fx is the focal length in world units (mm), w is the width in pixels,
	W is the width in world units (mm). so Z_ref * w/W will give us Z_ref in pixels units. we can assign w/W as s constant
	so the equation becomes f-n = Z_ref * s

	solving for both Z_ref = (-2*f*n)/(f+n) and f-n = Z_ref * s will give us the following
	f = 0.5 * ((Z_ref ** 2 + (Z_ref*s) ** 2)**0.5 + Z_ref + Z_ref*s) 
	n = 0.5 * ((Z_ref ** 2 + (Z_ref*s) ** 2)**0.5 + Z_ref - Z_ref*s)
	we can simplfy those to
	f = 0.5 * ((s**2 + 1)**0.5 + 1 + s) * Z_ref
	n = 0.5 * ((s**2 + 1)**0.5 + 1 - s) * Z_ref
	Now if we pluged that in our alpha and beta equation we will get
	alpha = ((s**2 + 1)**0.5 + 1)/s
	beta = -0.5 * ((2*(s**2 + 1)**0.5 + 2)/s) * Z_ref

	To be clear the s value is constant since the width and the height of a camera doesnt usually change
	it is only the Z_ref that will change from operation to the other. we want to make the projection matrix constant
	to make the computation faster and simpler so we have to get rid of Z_ref in beta. There is a trick to get
	rid of that Z_ref in beta. if we made beta = -0.5 * ((2*(s**2 + 1)**0.5 + 2)/s) and made the W value in
	the camera homogeneous coordiantes to be equal to Z_ref. the resulting (x, y, z, w) normlized device coordinates
	will be the same. so now the operation becomes.
							[[  fx/u0  ,    0    ,    0    ,    0    ],
			[x, y, z, w]=	 [    0    ,  fy/v0  ,    0    ,    0    ], * [X, Y, Z, Z_ref]
							 [    0    ,    0    ,  alpha  ,  beta   ],
							 [    0    ,    0    ,    1    ,    0    ]]	

	where alpha = ((s**2 + 1)**0.5 + 1)/s, and beta = -0.5 * ((2*(s**2 + 1)**0.5 + 2)/s)

	so now we just need to make our W for camera homogeneous to equal to Z_ref for the operation to be successful
	and since [X, Y, Z, Z_ref] is the same as [X/Z_ref, Y/Z_ref, Z/Z_ref, 1] after converting from homogeneous coordiantes
	to cartesian coordinates we can just simply devide our camera coordinates by the z reference point and get the same
	results.
	points: (Nx3) the 3D camera coordinates ([[x,y,z]]*N) which will be converted to normlized device coordinates
	z_ref: (Nx1) the Z reference point which will be used as the center for the created frustum (the depth center for
	device coordinates). 
	"""
	def camera_to_normlized_device(self, points, z_ref, check_points_outside = True):
		points = self.checks(points, check_points_dims=True, is_3d=True)
		z_ref = self.checks(z_ref, check_points_dims=True, is_1d=True)

		points = self.operation(matrix = self.projection_matrix,
								points = points/z_ref)

		if check_points_outside:
			condition_inside_negative = -1 <= points
			condition_inside_positive = points <= 1
			condition_inside = be.logical_and(condition_inside_negative, condition_inside_positive)
			condition_outside = be.logical_not(condition_inside)
			if be.any(condition_outside):
				raise PointOutsideDeviceError()
		return points


	"""
	http://kgeorge.github.io/2014/03/08/calculating-opengl-perspective-matrix-from-opencv-intrinsic-matrix
	https://github.com/anibali/pose3d-utils/blob/master/pose3d_utils/skeleton_normaliser.py#L8
	http://www.songho.ca/opengl/gl_projectionmatrix.html

	converting from normlized device coordinates to screen coordinates (image coordinates)
	since the coordinates now are normlized between -1 and +1 we can scale that to an image
	coordinates. first we need to peform a translation then multiplication.
	we can get a matrix to do that for us by doing A * P^-1 where A is the intrinsic[3x4]
	and P^-1 is the inverse of the projection matrix which would give us the following

		[[ u0 , 0  , 0 , u0 ],    [[ fx , 0  , u0 , 0  ],   |[[fx/u0 ,  0   ,  0   ,  0   ],|^-1
		 [ 0  , v0 , 0 , v0 ], =   [ 0  , fy , v0 , 0  ], * | [  0   ,fy/v0 ,  0   ,  0   ],|
		 [ 0  , 0  , 0 , 1  ]]     [ 0  , 0  , 1  , 0  ]]   | [  0   ,  0   ,alpha , beta ],|
															| [  0   ,  0   ,  1   ,  0   ]]|
		 
	where u0 and v0 are the principal point of the camera. this matrix project the coordinates
	to the size of the image size of the camera. we can change the screen size by scaling the
	the u0 and v0 according to the width and the height in the camera size units. the resulting
	matrix operation will become 
				[[u0*w/i_x,    0   ,   0  ,u0*w/i_x],
	[x, y, w] =  [    0   ,v0*h/i_y,   0  ,v0*h/i_y], * [X, Y, Z, 1]
				 [    0   ,    0   ,   0  ,    1   ]]
	
	where (x, y, w) is the 2D image homogeneous coordinate, (X, Y, Z, 1) is the 3D normlized
	device homogeneous coordinates, w is the width, h is the height, i_x and i_y is the height
	and the width of the camera image respectively
	
	However, we can extend this concept further. since what we do here is normlizing a data from [-1,+1]
	to [0, width]( x*(width/2) + width/2  ), normlizing the z axis would not heart.

	so the final matrix becomes
				   [[u0*w/i_x,    0   ,    0   ,u0*w/i_x],
	[x, y, z, w] =  [    0   ,v0*h/i_y,    0   ,v0*h/i_y], * [X, Y, Z, 1]
					[    0   ,    0   , z_size , z_size ],
					[    0   ,    0   ,    0   ,    1   ]]

	points: (Nx3) the 3D normlized device coordinates ([[x,y,z]]*N) which will be converted to screen coordinates
	width: (1) the width of the wanted screen in pixels
	height: (1) the height of the wanted screen in pixels
	depth: (1) the depth of the wanted screen in pixels
	"""
	def normlized_device_to_screen(self, points, width = None, height = None, depth = None):
		points = self.checks(points, check_points_dims=True, is_3d=True)

		width = self.size[0] if width is None else width
		height = self.size[1] if height is None else height
		depth = be.max(self.size) if depth is None else depth

		u0_f = width / self.size[0]
		v0_f = height / self.size[1]
		k0_f = depth / be.max(self.size)

		x_half_screen = u0_f * self.principal_point[0]
		y_half_screen = v0_f * self.principal_point[1]
		z_half_screen = k0_f * be.max(self.size)/2

		screen_matrix = [[x_half_screen,      0      ,      0      ,x_half_screen],
						 [      0      ,y_half_screen,      0      ,y_half_screen],
						 [      0      ,      0      ,z_half_screen,z_half_screen],
						 [      0      ,      0      ,      0      ,      1      ]]

		points = self.operation(matrix = screen_matrix, points = points)
		return points

	"""
	thanks to matrix properties we inverse this function normlized_device_to_screen by just
	inversing the matrix. However, width, height, and depth should be passed with the same
	values that normlized_device_to_screen got
	
	points: (Nx3) the screen coordinates ([[x,y,z]]*N) which will be converted to normlized device coordinates
	width: (1) the width of the encoded screen in pixels
	height: (1) the height of the encoded screen in pixels
	depth: (1) the depth of the encoded screen in pixels
	"""
	def screen_to_normlized_device(self, points, width = None, height = None, depth = None):
		points = self.checks(points, check_points_dims=True, is_3d=True)

		width = self.size[0] if width is None else width
		height = self.size[1] if height is None else height
		depth = be.max(self.size) if depth is None else depth

		u0_f = width / self.size[0]
		v0_f = height / self.size[1]
		k0_f = depth / be.max(self.size)

		x_half_screen = u0_f * self.principal_point[0]
		y_half_screen = v0_f * self.principal_point[1]
		z_half_screen = k0_f * be.max(self.size)/2

		screen_matrix = [[x_half_screen,      0      ,      0      ,x_half_screen],
						 [      0      ,y_half_screen,      0      ,y_half_screen],
						 [      0      ,      0      ,z_half_screen,z_half_screen],
						 [      0      ,      0      ,      0      ,      1      ]]

		points = self.operation(matrix = be.inverse(screen_matrix), points = points)
		return points


	"""
	inversing the function camera_to_normlized device is a little bit tricky since we lost the
	z_ref data in the points after applying that function, so z_ref data (camera coordinates) must be passed
	in order to get the inverse function correctly

	points: (Nx3) the 3D camera coordinates ([[x,y,z]]*N) which will be converted to normlized device coordinates
	z_ref: (Nx1) the Z reference point which will be used as the center for the created frustum (the depth center for
	device coordinates). 	
	"""
	def normlized_device_to_camera(self, points, z_ref):
		points = self.checks(points, check_points_dims=True, is_3d=True)
		z_ref = self.checks(z_ref, check_points_dims=True, is_1d=True)

		points = self.operation(matrix = be.inverse(self.projection_matrix),
								points = points)*z_ref
		return points


	"""
	converting from camera coordinates to world coordinates.
	it is the inverse function of world_to_camera which means
	(rotation_matrix^-1 * (points - translation_matrix))
	or it can be viewed as a matrix dot product with extrinsic_inverse matrix

	[Xw, Yw, Zw, Ww] = [[rotation_matrix(3x3), translation_matrix(1x3)],  ^-1   * [Xc, Yc, Zc, Wc] 
					   [	    0(3x1)        ,            1           ]]

	where (Xc, Yc, Zc, Wc) is the homogeneous coordinates of the camera and (Xw, Yw, Zw, Ww) is
	the homogeneous coordinates of the world. Note that the value of Wc doesnt matter but we will use
	1 so it is easier.

	points: (Nx3) the 3D canera coordinates ([[x,y,z]]*N) which will be converted to world coordinates
	"""
	def camera_to_world(self, points):
		# from slide 16 to 33 in [7] they clearly explained
		# the operation of converting from worlds coords to camera coords but they used
		# they used C translation matrix which is just - T translation matrix (our matrix here)
		# anyway mutlplying the inverse of the extrinsic matrix to the points will do the trick
		# since this operation is the inverse of the world_to_camera function.

		points = self.checks(points, check_points_dims=True, is_3d=True)
		points = self.operation(matrix = be.inverse(self.extrinsic), points = points)
		return points


if __name__ == '__main__':

	def CameraExtractor(cal_file_path):
		with open(cal_file_path,'r') as fp:
			camcalib_text = fp.read()
		new_data = []
		pattern = (
			r'name\s+(?P<name>\d+)\n  sensor\s+(?P<sensor>\d+ \d+)\n  size\s+(?P<size>\d+ '
			r'\d+)\n ' +
			r' animated\s+(?P<animated>\d)\n\s+intrinsic\s+(?P<intrinsic>(-?[0-9.]+\s*?){' +
			r'16})\s*\n  extrinsic\s+(?P<extrinsic>(-?[0-9.]+\s*?){16})\s*\n  radial\s+(' +
			r'?P<radial>\d)')
		new_data = [m.groupdict() for m in re.finditer(pattern, camcalib_text)]
		camera_list = list()
		for dict_camera in new_data:
			camera_name=int(dict_camera['name'])
			extrinsic = np.array(dict_camera['extrinsic'].split(' '), dtype=np.float32).reshape(4,4)
			intrinsic = np.array(dict_camera['intrinsic'].split(' '), dtype=np.float32).reshape(4,4)[:3]
			size = np.array(dict_camera['size'].split(' '), dtype=np.float32)
			camera_list.append(Camera(extrinsic, intrinsic, size))

		return camera_list

	import scipy.io
	np.set_printoptions(suppress=True)
	CG=CameraExtractor('D:/3DposeE/mpi_inf_3dhp/tools/camera.calibration')
	cam=CG[0]
	import time
	s = [-1286.56 ,   212.391,  1197.07 ]
	
	import matplotlib.pyplot as plt

	n_samples = 512*4*28
	random_points = np.random.uniform(-100,100,size=(n_samples,3)) + s
	random_z_ref = np.random.uniform(-100,100,size=(n_samples,1)).astype(np.float32) + s[-1]
	cdn_points = np.random.uniform(-1,1,size=(n_samples,3)).astype(np.float32)

	random_points = random_points.astype(np.float32)
	random_z_ref = random_z_ref.astype(np.float32)
	cdn_points = cdn_points.astype(np.float32)

	s_point = np.array([s])
	print('s_points')
	world_points = cam.camera_to_world(s_point)
	print('camera_to_world', world_points)
	cam_points = cam.world_to_camera(world_points)
	print('world_to_camera', cam_points)
	cdn_points = cam.camera_to_normlized_device(cam_points, cam_points[:,2:3], check_points_outside=False)
	print('camera_to_normlized_device', cdn_points)
	screen_points = cam.normlized_device_to_screen(cdn_points)
	print('normlized_device_to_screen', screen_points)
	cdn_points = cam.screen_to_normlized_device(screen_points)
	print('screen_to_normlized_device', cdn_points)
	cam_points = cam.normlized_device_to_camera(cdn_points, cam_points[:,2:3])
	print('camera_to_world', cam_points)
	world_points = cam.camera_to_world(cam_points)
	# print(cdn_points_1_operation[0])
	# exit()

	# cam_points = cam.world_to_camera(random_points)
	# cdn_points = cam.camera_to_normlized_device(random_points, random_z_ref, check_points_outside=False)
	# screen_points = cam.normlized_device_to_screen(cdn_points)

	#@tf.function
	exit()
	def func():
		cam_points = cam.world_to_camera(random_points)
		cdn_points = cam.camera_to_normlized_device(random_points, random_z_ref, check_points_outside=False)
		screen_points = cam.normlized_device_to_screen(cdn_points)
	func()
	start_time = time.perf_counter()
	#for i in range(1000):
	func()
	end_time = time.perf_counter()

	print(end_time - start_time)