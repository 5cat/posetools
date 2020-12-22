
class PoseToolsError(Exception):
	# for easier debugging
	def __init__(self, *args, **kwargs):
		self.args = args
		self.kwargs = kwargs
		super().__init__(*self.args, **self.kwargs) 

	def __str__(self):
		return self.message


class PointShapeError(PoseToolsError):
	#This exception raise when the wrong point dimension is fed
	#for exmaple feeding 3D points to a 2D points input, or feeding
	#2D points to a 1D point input etc.

	def __init__(self, current_n_cols, excpected_n_cols):
		self.current_n_cols = current_n_cols
		self.excpected_n_cols = excpected_n_cols
		self.message = f"The points should be {excpected_n_cols}D in the shape of " \
		f"(Nx{excpected_n_cols}) but instead got a (Nx{current_n_cols}) array"
		super().__init__(self.message)

class PointRankError(PoseToolsError):
	#This exception raise when the rank of the array is not the same as expected
	#because in the code all points are flattened to form (NxD) where N is the number
	#of points and D is the dimension, it is important to ensure all values are in the
	#same rank for faster numpy operations
	def __init__(self, current_rank, excpected_rank, array_shape):
		self.current_rank = current_rank
		self.excpected_rank = excpected_rank
		self.array_shape = array_shape
		self.message = f"The points should be in the shape of " \
		f"(NxD) with a rank of {excpected_rank} but instead got an "\
		f"array wit the shape of {array_shape} and a rank {current_rank} array"
		super().__init__(self.message)


class PointOutsideDeviceError(PoseToolsError):
	#when converting to normlized device coordinates, some points might fall outside
	#the frustum, this exception raise so it can be handeled by another class or function
	def __init__(self):
		self.message = "Points are outside of the clipped space."
		super().__init__(self.message)

class CameraParameterShapeError(PoseToolsError):
	def __init__(self, parameter_name, parameter_shape, parameter_expected_shape):
		self.parameter_name = parameter_name
		self.parameter_shape = parameter_shape
		self.parameter_expected_shape = parameter_expected_shape
		self.message = f"Expected parameter '{parameter_name}' to have the shape of {parameter_expected_shape}, "\
					   f"instead got a shape of {parameter_expected_shape}"
		super().__init__(self.message)

class CoordinateSystemTransformError(PoseToolsError):
	def __init__(self, current_coordinate_system, allowed_coordinate_system_transform):
		self.current_coordinate_system = current_coordinate_system
		self.allowed_coordinate_system_transform = allowed_coordinate_system_transform
		self.message = f"cannot transform from {current_coordinate_system} coordinate system to {allowed_coordinate_system_transform} coordinate system, "\
					   "using this function"
		super().__init__(self.message)

# class NormlizePoseError(PoseToolsError):
# 	def __init__(self,)

