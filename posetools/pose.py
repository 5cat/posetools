import scipy.io
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from .camera import Camera
from .exceptions import CoordinateSystemTransformError
from . import backend as be
class PoseExtractor:
	def __init__(self, path2mat , camera_list, camera_reference_index,
					 joint_names_original, joint_names_extract):

		data_raw = scipy.io.loadmat(path2mat)
		self.joint_names_original = joint_names_original
		self.joint_names_extract = joint_names_extract
		self.joint_name_extract_indices = [self.joint_names_original.index(i) for i in self.joint_names_extract]
		self.data = list()

		cam = camera_list[camera_reference_index]
		last_frame_index = -1
		for (frame_index,) in data_raw['frames']:
			if (frame_index-last_frame_index) != 1:
				raise Exception('Missing frame from the annot.mat file')

			camera_coords_3d = be.array(data_raw['annot3'][camera_reference_index][0][frame_index])
			camera_coords_3d = be.reshape(camera_coords_3d,(len(self.joint_names_original),3))
			camera_coords_3d = be.gather(camera_coords_3d, self.joint_name_extract_indices, axis = 0)
			world_coords_3d = cam.camera_to_world(camera_coords_3d)

			self.data.append(world_coords_3d)
			last_frame_index = frame_index

		self.max_frames = len(self.data)
		self.data = be.array(self.data)
		self.frame_indices = be.range(len(self.data))

	def __len__(self):
		return self.max_frames

	def __getitem__(self, key):
		return self.data[key]



class PoseSequence:
	def __init__(self, joint_names, joint_tree, poses, frame_indices, coordinate_system,
					name = None, camera = None, z_joint_reference_name = None,
					z_joint_reference_coordinates = None,
					screen_size = None):
		self.joint_names = joint_names
		self.joint_tree = joint_tree
		self.poses = poses
		self.frame_indices = frame_indices
		self.coordinate_system = coordinate_system
		self.name = name
		self.camera = camera
		self.z_joint_reference_name = z_joint_reference_name
		self.z_joint_reference_coordinates = z_joint_reference_coordinates
		self.screen_size = screen_size
		self.allowed_coordinate_systems = ['world', 'camera', 'normlized_device', 'screen']

		if self.coordinate_system not in self.allowed_coordinate_systems:
			raise TypeError(f"coordinate system not understood, please choose from the following {self.allowed_coordinate_systems}")

		if self.coordinate_system in self.allowed_coordinate_systems[1:]:
			if self.camera is None:
				raise TypeError(f"if coordinate system is one of the following {self.allowed_coordinate_systems[1:]}, "\
								"camera parameter must be passed")

		if self.coordinate_system in self.allowed_coordinate_systems[2:]:
			if self.z_joint_reference_name is None:
				raise TypeError(f"if the coordinate system is one of the following {self.allowed_coordinate_systems[2:]}, "\
								 "then z_joint_reference_name parameter must be passed")

		if self.coordinate_system == "screen":
			if self.screen_size is None:
				raise TypeError(f"if the camera coordinate_system is 'screen, then width, height, and depth parameter"\
								"must be passed")

		if self.poses.shape[1:] != (len(self.joint_names), 3):
			raise TypeError("poses should be in the shape of (PxJx3), where P is the number of poses and J is the number of joints. "\
							"make sure len(joint_names)==J")

		if len(self.joint_names)!=len(self.joint_tree):
			raise TypeError("joint_names length must equal joint_tree length")

	def poses_from_world_to_camera(self, camera = None):
		if self.coordinate_system != 'world':
			raise CoordinateSystemTransformError(self.coordinate_system, 'world')

		if camera is None:
			if self.camera is None:
				raise TypeError(f"if the coordinate system is 'camera', "\
								 "then camera parameter must be passed")
		else:
			self.camera = camera

		poses_shape = self.poses.shape
		poses_flatten = be.reshape(self.poses, (poses_shape[0]*poses_shape[1], poses_shape[2]))
		transformed_poses = self.camera.world_to_camera(poses_flatten)
		self.poses = be.reshape(transformed_poses, (poses_shape[0], poses_shape[1], poses_shape[2]))
		self.coordinate_system = 'camera'

	def poses_from_camera_to_normlized_device(self, z_joint_reference_name = None):
		if self.coordinate_system != 'camera':
			raise CoordinateSystemTransformError(self.coordinate_system, 'camera')

		if z_joint_reference_name is None:
			if self.z_joint_reference_name is None:
				raise TypeError(f"if the coordinate system is 'normlized_device', "\
								 "then z_joint_reference_name parameter must be passed")
		else:
			self.z_joint_reference_name = z_joint_reference_name

		z_joint_reference_index = self.joint_names.index(self.z_joint_reference_name)
		self.z_joint_reference_coordinates = self.poses[:, z_joint_reference_index:z_joint_reference_index+1, 2:3]

		poses_shape = self.poses.shape
		poses_flatten = be.reshape(self.poses, (poses_shape[0]*poses_shape[1], poses_shape[2]))

		z_joint_reference_coordinates = be.repeat(self.z_joint_reference_coordinates, poses_shape[1], axis = 1)
		z_joint_ref_coord_shape = z_joint_reference_coordinates.shape
		z_joint_reference_coordinates_flatten = be.reshape(z_joint_reference_coordinates,
											(z_joint_ref_coord_shape[0]*z_joint_ref_coord_shape[1], z_joint_ref_coord_shape[2]))

		transformed_poses = self.camera.camera_to_normlized_device(poses_flatten, z_joint_reference_coordinates_flatten,
											check_points_outside = False)
		self.poses = be.reshape(transformed_poses, (poses_shape[0], poses_shape[1], poses_shape[2]))
		self.coordinate_system = 'normlized_device'

	def poses_from_normlized_device_to_screen(self, screen_size = None):
		if self.coordinate_system != 'normlized_device':
			raise CoordinateSystemTransformError(self.coordinate_system, 'normlized_device')

		if screen_size is None:
			if self.screen_size is None:
				raise TypeError(f"if the coordinate system is 'screen', "\
								 "then screen_size parameter must be passed")
		else:
			self.screen_size = screen_size

		poses_shape = self.poses.shape
		poses_flatten = be.reshape(self.poses, (poses_shape[0]*poses_shape[1], poses_shape[2]))
		width = self.screen_size[0]
		height = self.screen_size[1]
		depth = self.screen_size[2]
		transformed_poses = self.camera.normlized_device_to_screen(poses_flatten,
									width = width, height = height, depth = depth)
		self.poses = be.reshape(transformed_poses, (poses_shape[0], poses_shape[1], poses_shape[2]))
		self.coordinate_system = 'screen'

	def poses_from_screen_to_normlized_device(self, screen_size = None):
		if self.coordinate_system != 'screen':
			raise CoordinateSystemTransformError(self.coordinate_system, 'screen')

		if screen_size is None:
			if self.screen_size is None:
				raise TypeError(f"if the coordinate system is 'screen', "\
								 "then screen_size parameter must be passed")
		elif self.screen_size is not None:
			if screen_size != self.screen_size:
				raise TypeError("screen_size parameter has already been passed, "\
								"changing the screen_size parameter will not give the correct result of this function")
		else:
			self.screen_size = screen_size

		poses_shape = self.poses.shape
		poses_flatten = be.reshape(self.poses, (poses_shape[0]*poses_shape[1], poses_shape[2]))
		width = self.screen_size[0]
		height = self.screen_size[1]
		depth = self.screen_size[2]
		transformed_poses = self.camera.screen_to_normlized_device(poses_flatten,
									width = width, height = height, depth = depth)
		self.poses = be.reshape(transformed_poses, (poses_shape[0], poses_shape[1], poses_shape[2]))
		self.coordinate_system = 'normlized_device'

	def poses_from_normlized_device_to_camera(self, z_joint_reference_coordinates = None):
		if self.coordinate_system != 'normlized_device':
			raise CoordinateSystemTransformError(self.coordinate_system, 'normlized_device')

		if z_joint_reference_coordinates is None:
			if self.z_joint_reference_coordinates is None:
				raise TypeError(f"if the coordinate system is 'normlized_device', "\
								 "then z_joint_reference_coordinates parameter must be passed")
		else:
			self.z_joint_reference_coordinates = z_joint_reference_coordinates

		poses_shape = self.poses.shape
		poses_flatten = be.reshape(self.poses, (poses_shape[0]*poses_shape[1], poses_shape[2]))

		z_joint_reference_coordinates = be.repeat(self.z_joint_reference_coordinates, poses_shape[1], axis = 1)
		z_joint_ref_coord_shape = z_joint_reference_coordinates.shape
		z_joint_reference_coordinates_flatten = be.reshape(z_joint_reference_coordinates,
											(z_joint_ref_coord_shape[0]*z_joint_ref_coord_shape[1], z_joint_ref_coord_shape[2]))

		transformed_poses = self.camera.normlized_device_to_camera(poses_flatten, z_joint_reference_coordinates_flatten)
		self.poses = be.reshape(transformed_poses, (poses_shape[0], poses_shape[1], poses_shape[2]))
		self.coordinate_system = 'camera'

	def poses_from_camera_to_world(self):
		if self.coordinate_system != 'camera':
			raise CoordinateSystemTransformError(self.coordinate_system, 'camera')

		if self.camera is None:
			raise TypeError("camera parameter should have been defined previously, "\
							"converting to a world coordinates from camera coordinates without knowing the camera "\
							"is not possible. make sure you have passed the camera parameter before")
		poses_shape = self.poses.shape
		poses_flatten = be.reshape(self.poses, (poses_shape[0]*poses_shape[1], poses_shape[2]))
		transformed_poses = self.camera.camera_to_world(poses_flatten)
		self.poses = be.reshape(transformed_poses, (poses_shape[0], poses_shape[1], poses_shape[2]))
		self.coordinate_system = 'world'

	def normlize_poses(self, camera = None, z_joint_reference_name = None, screen_size = None):

		if self.coordinate_system == 'world':
			self.poses_from_world_to_camera(camera = camera)

		if self.coordinate_system == 'camera':
			self.poses_from_camera_to_normlized_device(z_joint_reference_name = z_joint_reference_name)

		if self.coordinate_system == 'normlized_device':
			self.poses_from_normlized_device_to_screen(screen_size = screen_size)

	def denormlize_poses(self, screen_size = None, z_joint_reference_coordinates = None):
		if self.coordinate_system == 'screen':
			self.poses_from_screen_to_normlized_device(screen_size = screen_size)

		if self.coordinate_system == 'normlized_device':
			self.poses_from_normlized_device_to_camera(z_joint_reference_coordinates = z_joint_reference_coordinates)

		if self.coordinate_system == 'camera':
			self.poses_from_camera_to_world()



if __name__ == '__main__':
	print("start")
	joint_names_original = [
	# 0-3
	'spine3', 'spine4', 'spine2', 'spine',
	# 4-7
	'pelvis', 'neck', 'head', 'head_top',
	# 8-11
	'left_clavicle', 'left_shoulder', 'left_elbow', 'left_wrist',
	# 12-15
	'left_hand',  'right_clavicle', 'right_shoulder', 'right_elbow',
	# 16-19
	'right_wrist', 'right_hand', 'left_hip', 'left_knee',
	# 20-23
	'left_ankle', 'left_foot', 'left_toe', 'right_hip',
	# 24-27
	'right_knee', 'right_ankle', 'right_foot', 'right_toe'
	]

	joint_names_extract = [
	'head',
	'left_hand', 'right_hand',
	'pelvis',
	'left_knee','right_knee',
	'left_ankle','right_ankle'
	]

	joint_tree_dict = {
			"head": "head",
			"left_hand": "head",
			"right_hand": "head",
			"pelvis": "head",
			"left_knee": "pelvis",
			"left_ankle": "left_ankle",
			"right_knee": "pelvis",
			"right_ankle": "right_ankle"
	}

	joint_tree = [joint_names_extract.index(joint_tree_dict[i]) for i in joint_names_extract]

	import numpy as np
	import re
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
	#from tqdm import tqdm
	print("extract")
	path2test='D:/3DposeE/mpi_inf_3dhp/data/S{}/Seq{}/{}'
	camera_list = CameraExtractor(path2test.format(1,1,'camera.calibration'))
	pose_raw = PoseExtractor(path2test.format(1,1,'annot.mat'), camera_list, 0,
							joint_names_original, joint_names_extract)
	pose_sequence = PoseSequence(joint_names_extract, joint_tree, pose_raw.data, pose_raw.frame_indices, 'world',
								camera = camera_list[0], z_joint_reference_name = 'head',
								screen_size = (2048, 2048, 2048))
	import time

	print("prerun")
	pose_sequence.normlize_poses()
	pose_sequence.denormlize_poses()

	
	def func():
		pose_sequence = PoseSequence(joint_names_extract, joint_tree, pose_raw.data, pose_raw.frame_indices, 'world',
									camera = camera_list[0], z_joint_reference_name = 'head',
									screen_size = (2048, 2048, 2048))
		
		pose_sequence.normlize_poses()
		pose_sequence.denormlize_poses()

		start_time = time.perf_counter()
		for _ in range(1):
			pose_sequence.normlize_poses()
			pose_sequence.denormlize_poses()

		end_time = time.perf_counter()
		return end_time - start_time

	print("run")
	print(func())
	# for i in tqdm(range(8)):
	# 	for j in range(2):
	# 		AE = PoseExtractor(path2test.format(i+1,j+1),CameraExtractor('camera.calibration'),
	# 			joint_names,[joint_names.index(i) for i in joint_names_data])
