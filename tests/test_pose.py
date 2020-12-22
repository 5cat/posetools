import unittest
from posetools import camera, pose, exceptions
from posetools import backend as be

class TestPose(be.unittest_backend):

	@staticmethod
	def repeat_array(array, n_repeats=5, axis=0):
		return be.repeat(be.array([array]), n_repeats, axis=axis)

	@classmethod
	def setUpClass(cls):
		cls.joint_names = [
									'head',
									'left_hand', 'right_hand',
									'pelvis',
									'left_knee','right_knee',
									'left_ankle','right_ankle'
									]
		cls.joint_tree_dict = {
							"head": "head",
							"left_hand": "head",
							"right_hand": "head",
							"pelvis": "head",
							"left_knee": "pelvis",
							"left_ankle": "left_ankle",
							"right_knee": "pelvis",
							"right_ankle": "right_ankle"
						  }
		cls.joint_tree = [cls.joint_names.index(cls.joint_tree_dict[i]) for i in cls.joint_names]


		extrinsic = [[0.9650164, 0.00488022, 0.262144, -562.8666],
					 [-0.004488356, -0.9993728, 0.0351275, 1398.138],
					 [0.262151, -0.03507521, -0.9643893, 3852.623],
					 [0.0, 0.0, 0.0, 1.0]]
		intrinsic = [[1497.693, 0.0, 1024.704, 0.0],
					 [0.0, 1497.103, 1051.394, 0.0],
					 [0.0, 0.0, 1.0, 0.0]]
		size = (2048,2048)
		cls.camera0 = camera.Camera(extrinsic, intrinsic, size)

		cls.point0_3d_camera = [
		[11.99029541015625, -164.093017578125, 3696.260009765625],
		[-57.21343994140625, 51.520751953125, 4469.81982421875],
		[74.39117431640625, 90.62548828125, 2930.449951171875],
		[1.67559814453125, 402.92889404296875, 3713.26025390625],
		[-45.7490234375, 956.8289794921875, 3800.59033203125],
		[-15.40264892578125, 957.8069458007812, 3670.330078125],
		[-137.36196899414062, 1388.239990234375, 3780.199951171875],
		[-87.24114990234375, 1390.7698974609375, 3718.400146484375]]
		cls.point0_3d_camera = cls.repeat_array(cls.point0_3d_camera)

		cls.point0_normlized_device =  [[0.004741237964481115, -0.06321407109498978, 0.0],
										[-0.01870821602642536, 0.016412636265158653, 0.3408827781677246],
										[0.037103209644556046, 0.044035423547029495, -0.5147398710250854],
										[0.0006595365121029317, 0.15451093018054962, 0.00901783350855112],
										[-0.017593618482351303, 0.3584837317466736, 0.054070405662059784],
										[-0.006133588496595621, 0.3715857267379761, -0.013915406540036201],
										[-0.05310998111963272, 0.5229208469390869, 0.043737415224313736],
										[-0.03429174795746803, 0.5325806140899658, 0.011728131212294102]]
		cls.point0_normlized_device = cls.repeat_array(cls.point0_normlized_device)

		cls.point0_screen = [[1029.5623449021107, 984.9311451876747, 1024.0],
							 [1005.5335960807099, 1068.6501909673507, 1373.06396484375],
							 [1062.723786066897, 1097.6926249657026, 496.9063720703125],
							 [1025.3798091807598, 1213.8459145341403, 1033.2342615127563],
							 [1006.6757286196548, 1428.30170302841, 1079.3680953979492],
							 [1018.4188669511584, 1442.0770625132718, 1009.7506237030029],
							 [970.2819704881449, 1601.1899063846795, 1068.7871131896973],
							 [989.5650888964269, 1611.3461280235788, 1036.0096063613892]]
		cls.point0_screen = cls.repeat_array(cls.point0_screen)

		cls.z_ref0 = [[3696.26]]
		cls.z_ref0 = cls.repeat_array(cls.z_ref0)


		extrinsic = [[0.6050639, -0.02184232, 0.7958773, -1429.856],
					 [-0.22647, -0.9630526, 0.1457429, 738.1779],
					 [0.7632883, -0.2684261, -0.587655, 4897.966],
					 [0.0, 0.0, 0.0, 1.0]]
		intrinsic = [[1495.217, 0.0, 1030.519, 0.0],
					 [0.0, 1495.52, 1052.626, 0.0],
					 [0.0, 0.0, 1.0, 0.0]]
		size = (2048,2048)
		cls.camera1 = camera.Camera(extrinsic, intrinsic, size)

		cls.point1_3d_camera = [[-952.7672119140625, -855.3687133789062, 4729.232421875],
								[-1467.8984375, -762.2444458007812, 5342.1279296875],
								[-425.6014404296875, -492.96514892578125, 4233.24853515625],
								[-944.58984375, -307.07574462890625, 4874.89404296875],
								[-1009.2080078125, 223.68951416015625, 5050.375],
								[-905.89697265625, 240.66323852539062, 4967.138671875],
								[-1049.097412109375, 661.887451171875, 5086.6455078125],
								[-971.677734375, 665.3311157226562, 5068.4228515625]]
		cls.point1_3d_camera = cls.repeat_array(cls.point1_3d_camera)

		cls.point1_normlized_device =  [[-0.2923104465007782, -0.25696906447410583, 0.0],
										[-0.3986850082874298, -0.20272070169448853, 0.22581380605697632],
										[-0.14587387442588806, -0.1654476523399353, -0.2306067794561386],
										[-0.28114238381385803, -0.08949495106935501, 0.058810945600271225],
										[-0.2899380624294281, 0.06292744725942612, 0.12515634298324585],
										[-0.26461881399154663, 0.06883694976568222, 0.09427087008953094],
										[-0.29924890398979187, 0.1848718822002411, 0.13829825818538666],
										[-0.2781619131565094, 0.18650184571743011, 0.13171915709972382]]
		cls.point1_normlized_device = cls.repeat_array(cls.point1_normlized_device)

		cls.point1_screen = [[729.287561391, 782.1336641240923, 1024.0],
							 [619.6665497823997, 839.2368999719038, 1255.2333374023438],
							 [880.1932375012402, 878.4714799482026, 787.8586578369141],
							 [740.79646266294, 958.421266295707, 1084.2224082946777],
							 [731.7323483537621, 1118.8650421865386, 1152.1600952148438],
							 [757.8243160226557, 1125.085538033185, 1120.5333709716797],
							 [722.1373488197423, 1247.2269221024762, 1165.617416381836],
							 [743.8678944323474, 1248.9426640415186, 1158.8804168701172]]
		cls.point1_screen = cls.repeat_array(cls.point1_screen)


		cls.z_ref1 = [[4729.2324]]
		cls.z_ref1 = cls.repeat_array(cls.z_ref1)


		cls.point_3d_world =   [[520.7674560546875, 1569.5408935546875, 246.61279296875],
								[655.806396484375, 1326.5919189453125, -509.96728515625],
								[379.0841979980469, 1342.147705078125, 1010.457275390625],
								[512.7252197265625, 1002.22802734375, 247.43212890625],
								[487.3672180175781, 445.380859375, 170.237060546875],
								[482.4997863769531, 449.1204833984375, 303.847900390625],
								[391.6775817871094, 14.508544921875, 181.039794921875],
								[423.8327331542969, 14.3924560546875, 253.86669921875]]
		cls.point_3d_world = cls.repeat_array(cls.point_3d_world)



		cls.frame_indices = be.range(5)


	def test_poses_from_world_to_camera(self):
		pose_seq_0 = pose.PoseSequence(self.joint_names, self.joint_tree,
			self.point_3d_world, self.frame_indices, 'world', camera = self.camera0)
		pose_seq_1 = pose.PoseSequence(self.joint_names, self.joint_tree,
			self.point_3d_world, self.frame_indices, 'world', camera = self.camera1)
		pose_seq_0.poses_from_world_to_camera()
		pose_seq_1.poses_from_world_to_camera()
		point0_3d_camera = pose_seq_0.poses
		point1_3d_camera = pose_seq_1.poses

		self.assertIsClose(self.point0_3d_camera, point0_3d_camera, atol = 1e-2, rtol = 0)
		self.assertIsClose(self.point1_3d_camera, point1_3d_camera, atol = 1e-2, rtol = 0)


	def test_poses_from_camera_to_normlized_device(self):
		pose_seq_0 = pose.PoseSequence(self.joint_names, self.joint_tree,
			self.point0_3d_camera, self.frame_indices, 'camera', camera = self.camera0, z_joint_reference_name = 'head')
		pose_seq_1 = pose.PoseSequence(self.joint_names, self.joint_tree,
			self.point1_3d_camera, self.frame_indices, 'camera', camera = self.camera1, z_joint_reference_name = 'head')
		pose_seq_0.poses_from_camera_to_normlized_device()
		pose_seq_1.poses_from_camera_to_normlized_device()
		point0_normlized_device = pose_seq_0.poses
		point1_normlized_device = pose_seq_1.poses

		self.assertIsClose(self.point0_normlized_device, point0_normlized_device, atol = 1e-6, rtol = 0)
		self.assertIsClose(self.point1_normlized_device, point1_normlized_device, atol = 1e-6, rtol = 0)

	def test_poses_from_normlized_device_to_screen(self):
		pose_seq_0 = pose.PoseSequence(self.joint_names, self.joint_tree,
			self.point0_normlized_device, self.frame_indices, 'normlized_device', camera = self.camera0, screen_size = (2048, 2048, 2048),
			z_joint_reference_name = 'head')
		pose_seq_1 = pose.PoseSequence(self.joint_names, self.joint_tree,
			self.point1_normlized_device, self.frame_indices, 'normlized_device', camera = self.camera1, screen_size = (2048, 2048, 2048),
			z_joint_reference_name = 'head')
		pose_seq_0.poses_from_normlized_device_to_screen()
		pose_seq_1.poses_from_normlized_device_to_screen()
		point0_screen = pose_seq_0.poses
		point1_screen = pose_seq_1.poses

		self.assertIsClose(self.point0_screen, point0_screen, atol = 1e-2, rtol = 0)
		self.assertIsClose(self.point1_screen, point1_screen, atol = 1e-2, rtol = 0)

	def test_poses_from_screen_to_normlized_device(self):
		pose_seq_0 = pose.PoseSequence(self.joint_names, self.joint_tree,
			self.point0_screen, self.frame_indices, 'screen', camera = self.camera0, screen_size = (2048, 2048, 2048),
			z_joint_reference_name = 'head')
		pose_seq_1 = pose.PoseSequence(self.joint_names, self.joint_tree,
			self.point1_screen, self.frame_indices, 'screen', camera = self.camera1, screen_size = (2048, 2048, 2048),
			z_joint_reference_name = 'head')
		pose_seq_0.poses_from_screen_to_normlized_device()
		pose_seq_1.poses_from_screen_to_normlized_device()
		point0_normlized_device = pose_seq_0.poses
		point1_normlized_device = pose_seq_1.poses

		self.assertIsClose(self.point0_normlized_device, point0_normlized_device, atol = 1e-6, rtol = 0)
		self.assertIsClose(self.point1_normlized_device, point1_normlized_device, atol = 1e-6, rtol = 0)

	def test_poses_from_normlized_device_to_camera(self):
		pose_seq_0 = pose.PoseSequence(self.joint_names, self.joint_tree,
			self.point0_normlized_device, self.frame_indices, 'normlized_device', camera = self.camera0,
			z_joint_reference_name = 'head', z_joint_reference_coordinates = self.z_ref0)
		pose_seq_1 = pose.PoseSequence(self.joint_names, self.joint_tree,
			self.point1_normlized_device, self.frame_indices, 'normlized_device', camera = self.camera1,
			z_joint_reference_name = 'head', z_joint_reference_coordinates = self.z_ref1)
		pose_seq_0.poses_from_normlized_device_to_camera()
		pose_seq_1.poses_from_normlized_device_to_camera()
		point0_3d_camera = pose_seq_0.poses
		point1_3d_camera = pose_seq_1.poses

		self.assertIsClose(self.point0_3d_camera, point0_3d_camera, atol = 1e-2, rtol = 0)
		self.assertIsClose(self.point1_3d_camera, point1_3d_camera, atol = 1e-2, rtol = 0)

	def test_poses_from_camera_to_world(self):
		pose_seq_0 = pose.PoseSequence(self.joint_names, self.joint_tree,
			self.point0_3d_camera, self.frame_indices, 'camera', camera = self.camera0)
		pose_seq_1 = pose.PoseSequence(self.joint_names, self.joint_tree,
			self.point1_3d_camera, self.frame_indices, 'camera', camera = self.camera1)

		pose_seq_0.poses_from_camera_to_world()
		pose_seq_1.poses_from_camera_to_world()
		point0_3d_world = pose_seq_0.poses
		point1_3d_world = pose_seq_1.poses

		self.assertIsClose(self.point_3d_world, point0_3d_world, atol = 1e-2, rtol = 0)
		self.assertIsClose(self.point_3d_world, point1_3d_world, atol = 1e-2, rtol = 0)

	def test_normlize_poses(self):
		pose_seq_0 = pose.PoseSequence(self.joint_names, self.joint_tree,
			self.point_3d_world, self.frame_indices, 'world', camera = self.camera0,
			screen_size = (2048, 2048, 2048), z_joint_reference_name = 'head')
		pose_seq_1 = pose.PoseSequence(self.joint_names, self.joint_tree,
			self.point_3d_world, self.frame_indices, 'world', camera = self.camera1,
			screen_size = (2048, 2048, 2048), z_joint_reference_name = 'head')

		pose_seq_0.normlize_poses()
		pose_seq_1.normlize_poses()
		point0_screen = pose_seq_0.poses
		point1_screen = pose_seq_1.poses

		self.assertIsClose(self.point0_screen, point0_screen, atol = 1e-2, rtol = 0)
		self.assertIsClose(self.point1_screen, point1_screen, atol = 1e-2, rtol = 0)


	def test_denormlize_poses(self):
		pose_seq_0 = pose.PoseSequence(self.joint_names, self.joint_tree,
			self.point0_screen, self.frame_indices, 'screen', camera = self.camera0,
			screen_size = (2048, 2048, 2048), z_joint_reference_name = 'head',
			z_joint_reference_coordinates = self.z_ref0)
		pose_seq_1 = pose.PoseSequence(self.joint_names, self.joint_tree,
			self.point1_screen, self.frame_indices, 'screen', camera = self.camera1,
			screen_size = (2048, 2048, 2048), z_joint_reference_name = 'head',
			z_joint_reference_coordinates = self.z_ref1)

		pose_seq_0.denormlize_poses()
		pose_seq_1.denormlize_poses()
		point0_3d_world = pose_seq_0.poses
		point1_3d_world = pose_seq_1.poses

		self.assertIsClose(self.point_3d_world, point0_3d_world, atol = 1e-2, rtol = 0)
		self.assertIsClose(self.point_3d_world, point1_3d_world, atol = 1e-2, rtol = 0)

	