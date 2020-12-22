import unittest
from posetools import camera, exceptions
from posetools import backend as be

class TestCamera(be.unittest_backend):

	@staticmethod
	def repeat_array(array, n_repeats=5, axis=0):
		return be.repeat(be.array([array]), n_repeats, axis=axis)

	@classmethod
	def setUpClass(cls):
		extrinsic = [[0.9650164, 0.00488022, 0.262144, -562.8666],
					 [-0.004488356, -0.9993728, 0.0351275, 1398.138],
					 [0.262151, -0.03507521, -0.9643893, 3852.623],
					 [0.0, 0.0, 0.0, 1.0]]
		intrinsic = [[1497.693, 0.0, 1024.704, 0.0],
					 [0.0, 1497.103, 1051.394, 0.0],
					 [0.0, 0.0, 1.0, 0.0]]
		size = (2048,2048)
		cls.camera0 = camera.Camera(extrinsic, intrinsic, size)

		cls.point0_3d_camera = [-1286.56 ,   212.391,  1197.07 ]
		cls.point0_3d_camera = cls.repeat_array(cls.point0_3d_camera)

		cls.point0_normlized_device = [-1.5708504915237427, 0.25264036655426025, -0.49845263361930847]
		cls.point0_normlized_device = cls.repeat_array(cls.point0_normlized_device)

		cls.point0_screen = [-584.953, 1317.02, -0.49845263361930847 * max(size)/2 +  max(size)/2 ]
		cls.point0_screen = cls.repeat_array(cls.point0_screen)

		extrinsic = [[0.6050639, -0.02184232, 0.7958773, -1429.856],
					 [-0.22647, -0.9630526, 0.1457429, 738.1779],
					 [0.7632883, -0.2684261, -0.587655, 4897.966],
					 [0.0, 0.0, 0.0, 1.0]]
		intrinsic = [[1495.217, 0.0, 1030.519, 0.0],
					 [0.0, 1495.52, 1052.626, 0.0],
					 [0.0, 0.0, 1.0, 0.0]]
		size = (2048,2048)
		cls.camera1 = camera.Camera(extrinsic, intrinsic, size)

		cls.point1_3d_camera = [-444.164,  164.796, 2126.44 ]
		cls.point1_3d_camera = cls.repeat_array(cls.point1_3d_camera)

		cls.point1_normlized_device = [-0.30306684970855713, 0.11010617017745972, 0.5798351764678955]
		cls.point1_normlized_device = cls.repeat_array(cls.point1_normlized_device)

		cls.point1_screen = [ 718.203, 1168.53, 0.5798351764678955 * max(size)/2 +  max(size)/2 ]
		cls.point1_screen = cls.repeat_array(cls.point1_screen)


		cls.point_3d_world = [-1389.21 ,  1274.61,  2329.62]
		cls.point_3d_world = cls.repeat_array(cls.point_3d_world)

		cls.z_ref = [1500]
		cls.z_ref = cls.repeat_array(cls.z_ref)


	def test_world_to_camera(self):
		point0_3d_camera = self.camera0.world_to_camera(self.point_3d_world)
		point1_3d_camera = self.camera1.world_to_camera(self.point_3d_world)

		self.assertIsClose(self.point0_3d_camera, point0_3d_camera, atol = 1e-2, rtol = 0)
		self.assertIsClose(self.point1_3d_camera, point1_3d_camera, atol = 1e-2, rtol = 0)

	def test_camera_to_normlized_device(self):
		point0_normlized_device = self.camera0.camera_to_normlized_device(
			self.point0_3d_camera, self.z_ref, check_points_outside = False)
		point1_normlized_device = self.camera1.camera_to_normlized_device(
			self.point1_3d_camera, self.z_ref, check_points_outside = False)

		self.assertIsClose(self.point0_normlized_device, point0_normlized_device, atol = 1e-6, rtol = 0)
		self.assertIsClose(self.point1_normlized_device, point1_normlized_device, atol = 1e-6, rtol = 0)

	def test_normlized_device_to_screen(self):
		point0_screen = self.camera0.normlized_device_to_screen(self.point0_normlized_device)
		point1_screen = self.camera1.normlized_device_to_screen(self.point1_normlized_device)

		self.assertIsClose(self.point0_screen, point0_screen, atol = 1e-2, rtol = 0)
		self.assertIsClose(self.point1_screen, point1_screen, atol = 1e-2, rtol = 0)

	def test_screen_to_normlized_device(self):
		point0_normlized_device = self.camera0.screen_to_normlized_device(self.point0_screen)
		point1_normlized_device = self.camera1.screen_to_normlized_device(self.point1_screen)

		self.assertIsClose(self.point0_normlized_device, point0_normlized_device, atol = 1e-5, rtol = 0)
		self.assertIsClose(self.point1_normlized_device, point1_normlized_device, atol = 1e-5, rtol = 0)

	def test_normlized_device_to_camera(self):
		point0_3d_camera = self.camera0.normlized_device_to_camera(self.point0_normlized_device, self.z_ref)
		point1_3d_camera = self.camera1.normlized_device_to_camera(self.point1_normlized_device, self.z_ref)

		self.assertIsClose(self.point0_3d_camera, point0_3d_camera, atol = 1e-2, rtol = 0)
		self.assertIsClose(self.point1_3d_camera, point1_3d_camera, atol = 1e-2, rtol = 0)

	def test_camera_to_world(self):
		point0_3d_world = self.camera0.camera_to_world(self.point0_3d_camera)
		point1_3d_world = self.camera1.camera_to_world(self.point1_3d_camera)

		self.assertIsClose(self.point_3d_world, point1_3d_world, atol = 1e-2, rtol = 0)
		self.assertIsClose(self.point_3d_world, point1_3d_world, atol = 1e-2, rtol = 0)

	def test_camera_extrinsic_CameraParameterShapeError(self):
		with self.assertRaises(exceptions.CameraParameterShapeError):
			camera.Camera(be.eye(1) ,be.eye(4)[:3], [0,0])

	def test_camera_intrinsic_CameraParameterShapeError(self):
		with self.assertRaises(exceptions.CameraParameterShapeError):
			camera.Camera(be.eye(4), be.eye(2), [0,0])

	def test_camera_size_CameraParameterShapeError(self):
		with self.assertRaises(exceptions.CameraParameterShapeError):
			camera.Camera(be.eye(4), be.eye(4)[:3], [5,6,3])

	def test_camera_distortion_coefficient_CameraParameterShapeError(self):
		with self.assertRaises(exceptions.CameraParameterShapeError):
			camera.Camera(be.eye(4), be.eye(4)[:3], [5,6], distortion_coefficient = be.ones(2))

	def test_camera_radial_CameraParameterShapeError(self):
		with self.assertRaises(exceptions.CameraParameterShapeError):
			camera.Camera(be.eye(4), be.eye(4)[:3], [5,6], radial = be.ones(2))

	def test_camera_to_normlized_device_PointOutsideDeviceError(self):
		with self.assertRaises(exceptions.PointOutsideDeviceError):
			self.camera0.camera_to_normlized_device(self.point0_3d_camera, self.z_ref)


if __name__ == '__main__':
	unittest.main()