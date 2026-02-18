"""
Geometry contains primitive functions for geometric operations on MoveableThings 
position and orientation. Orientations are currently only supported for (improper)
Euler angles and angles are currently assumed to be in radian. To convert angles the 
predefined constants ``DEGREES_TO_RADIANS`` and ``RADIANS_TO_DEGREES`` can be used.

Attributes
----------
TAU : float
	Tau is the circumference of a unit circle. :math:`\\pi \\times 2 = \\tau`.
PI : float
	Pi is the half circumference of a unit circle. :math:`\\pi \\times 2 = \\tau`.
DEGREES_TO_RADIANS : float
	To convert an angle in degrees to radian it can be multiplied with this constant.
RADIANS_TO_DEGREES : float
	To convert an angle in radian to degrees it can be multiplied with this constant.
"""
import numpy as np
import blueprints as blue



TAU = 6.2831853071795864769252867664
PI  = 3.1415926535897932384626433832

DEGREES_TO_RADIANS = TAU / 360
RADIANS_TO_DEGREES = 360 / TAU



class Rotation:
	"""
	This class implements multiple functions that operate on a MoveableThings orientation.
	
	Attributes
	----------
	X : np.ndarray
		The unit vector :math:`\\vec{\\text{id}_x}` representing the ``X``-axis.
	Y : np.ndarray
		The unit vector :math:`\\vec{\\text{id}_y}` representing the ``Y``-axis.
	Z : np.ndarray
		The unit vector :math:`\\vec{\\text{id}_z}` representing the ``Z``-axis.
	"""
	X = np.array([1, 0, 0])
	Y = np.array([0, 1, 0])
	Z = np.array([0, 0, 1])
	@classmethod
	def euler(cls, 
		  vector, 
		  alpha, 
		  beta, 
		  gamma):
		"""
		Rotates a vector according to improper euler angles.
		
		Parameters
		----------
		vector : np.ndarray
			The vector to be rotated.
		alpha : float
			The angle along which the vector is rotated on the ``X``-axis.
		beta : float
			The angle along which the vector is rotated on the ``Y``-axis.
		gamma : float
			The angle along which the vector is rotated on the ``Z``-axis.
		
		Returns
		-------
		np.ndarray
			The rotated vector.
		"""
		R = cls.E_rot(alpha, beta, gamma)
		return R @ vector


	@classmethod
	def rotate_around_axis(cls, 
			       vector, 
			       axis, 
			       angle):
		"""
		Rotates a vector around an axis for a given angles.
		
		Parameters
		----------
		vector : np.ndarray
			The vector to be rotated
		axis : np.ndarray
			The axis around which the vector is rotated according to the right hand rule.
		angle : float
			The angle by which the vector is rotated.
		
		Returns
		-------
		np.ndarray
			The rotated vector.
		"""
		axis = Vector.normalize(axis)
		return vector * np.cos(angle) + \
		np.cross(axis, vector) * np.sin(angle) + \
		axis * np.dot(axis, vector) * (1 - np.cos(angle))


	@classmethod
	def global_orientation(cls, node):
		"""
		This method calculates the global orientation of a Thing (typically a node in a kinematic tree).
		
		Parameters
		----------
		node : Thing
			A Thing (typically a node in a kinematic tree) for which the global orientation is calculated w.r.t it's root.
		
		Returns
		-------
		(float, float, float)
			The improper euler angles representing the global orientation.
		"""
		R = np.eye(3)
		while node is not None and isinstance(node, blue.MoveableThingType):
			M = cls.E_rot(node.alpha, node.beta, node.gamma)
			R = M @ R
			node = node.parent
		alpha, beta, gamma = cls.reference_frame_to_euler(R)
		return alpha, beta, gamma


	@classmethod
	def global_rotation_matrix(cls, node):
		"""
		This method calculates the rotation matrix for the global orientation of a Thing (typically a node in e kinematic tree) w.r.t. to its root.
		
		Parameters
		----------
		node : Thing
			A Thing (typically a node in a kinematic tree) for which the global orientation is calculated w.r.t it's root.
		
		Returns
		-------
		np.ndarray
			The global orientation matrix for the node.
		"""
		alpha, beta, gamma = cls.global_orientation(node)
		return cls.E_rot(alpha, beta, gamma)


	@classmethod
	def euler_to_reference_frame(cls, 
				     alpha: float, 
				     beta:  float, 
				     gamma: float):
		"""
		This method constructs a reference frame (i. e. the unit vectors for the axis of the local coordinate system) for (improper) euler angles.
		
		Parameters
		----------
		alpha : float
			The angle by which the ``X``-axis is rotated to reach the reference frame.
		beta : float
			The angle by which the ``Y``-axis is rotated to reach the reference frame.
		gamma : float
			The angle by which the ``Z``-axis is rotated to reach the reference frame.
		
		Returns
		-------
		(np.ndarray, np.ndarray, np.ndarray)
			A tuple of the three unit vectors for the given orientation representing the reference frame.
		"""
		M = cls.M_rot(alpha, beta, gamma)
		R_x, R_y, R_z = np.rollaxis(M, axis=1)
		return R_x, R_y, R_z


	@classmethod
	def reference_frame_to_euler(cls, R):
		"""
		Computes the (improper) euler angles that lead to the given reference frame.
		
		Parameters
		----------
		R : np.ndarray
			A matrix representing the reference frame as a concatenation of the unit coulmn vectors of the orientations coordinate system.
		
		Returns
		-------
		(float, float, float)
			The (improper) euler angles for the given reference frame.
		
		Raises
		------
		ArithmeticError
		If the reference frame does not contain orthogonal vectors that do not adhere to the right hand rule convention am ArithmeticError is raised.
		"""
		X = np.array([1, 0, 0])
		Y = np.array([0, 1, 0])
		Z = np.array([0, 0, 1])
		R_x, R_y, R_z = np.rollaxis(R, axis=1)
		try:
			assert Vector.equal(np.cross(R_x, R_y), R_z)
		except AssertionError:
			raise ArithmeticError('Referenceframe to euler convertion only supports rotation and not reflections but the given reference frame contains a reflected axis.')
		R_x = Vector.normalize(R_x)
		R_y = Vector.normalize(R_y)
		R_z = Vector.normalize(R_z)
		R_x1, R_x2, R_x3 = R_x
		R_y1, R_y2, R_y3 = R_y
		R_z1, R_z2, R_z3 = R_z
		if R_y1 != 0:
			Y_bar = np.array([0, -R_z3/R_y1, R_z2/R_y1])
			Y_bar = Vector.normalize(Y_bar)
			X_bar = np.cross(Y_bar, R_z)
		else:
			X_bar = R_x
			Y_bar = R_y
		Z_bar = np.cross(X, Y_bar)
		alpha = cls.angle(Z,     Z_bar, axis=X)
		beta  = cls.angle(X,     X_bar, axis=Y_bar)
		gamma = cls.angle(Y_bar, R_y,   axis=R_z)
		return alpha, beta, gamma


	@classmethod
	def angle(cls, a, b, axis=None) -> float:
		"""
		Computes the angle between two vectors. 
		If axis is set, the angle is measured according to a rotation 
		around axis from a to b, otherwise the shortest angle is returned.
		
		Parameters
		----------
		a : np.ndarray
			A vector for which the angle is computed w.r.t. b.
		b : np.ndarray
			A vector for which the angle is computed w.r.t. a.
		axis : None|np.ndarray, optional
			The axis of rotation can be specified optionally, if this is not done, the resulting angles will be in the range [0, PI).
		
		Returns
		-------
		float
			The resulting angle will be in the range [0, PI) if axis is not specified, otherwise the angle will be in the range [0, TAU).
		"""
		a     = Vector.normalize(a)
		b     = Vector.normalize(b)
		c     = np.cross(a, b)
		axis  = Vector.normalize(axis) if axis is not None else c
		sign  = np.sign(axis @ c)
		sign  = sign if sign != 0 else 1
		# DOT PRODUCT CLIPPING FOR NUMERICAL STABILITY
		angle = np.arccos(np.clip(a @ b, -1, 1))
		return sign * angle


	@classmethod
	def N_rot(cls, node=None) -> np.ndarray:
		"""
		This method returns the rotation matrix for a given Thing. If the Thing does not inherit
		from MoveableThing the identity metrix is returned.
		
		Parameters
		----------
		node : None, ThingType
			A Thing, typically a node from a kinematic tree.
		
		Returns
		-------
		np.ndarray
			A rotation matrix from the Things orientation.
		"""
		if node and isinstance(node, blue.MoveableThingType):
			alpha, beta, gamma = node.alpha, node.beta, node.gamma
		else:
			alpha, beta, gamma = 0, 0, 0
		return cls.E_rot(alpha, beta, gamma)


	@staticmethod
	def X_rot(alpha) -> np.ndarray:
		"""
		Constructs the rotation matrix around the ``X``-axis for a given angle.
		
		Parameters
		----------
		alpha : float
			The angle of rotation.
		
		Returns
		-------
		np.ndarray
			The rotation matrix for the given angle.
		"""
		return np.array([[             1,             0,             0],
				 [             0, np.cos(alpha),-np.sin(alpha)],
				 [             0, np.sin(alpha), np.cos(alpha)]])


	@staticmethod
	def Y_rot(beta) -> np.ndarray:
		"""
		Constructs the rotation matrix around the ``Y``-axis for a given angle.
		
		Parameters
		----------
		beta : float
			The angle of rotation.
		
		Returns
		-------
		np.ndarray
			The rotation matrix for the given angle.
		"""
		return np.array([[ np.cos(beta) ,             0, np.sin(beta) ],
				 [             0,             1,             0],
				 [-np.sin(beta) ,             0, np.cos(beta) ]])


	@staticmethod
	def Z_rot(gamma) -> np.ndarray:
		"""
		Constructs the rotation matrix around the ``Z``-axis for a given angle.
		
		Parameters
		----------
		gamma : float
			The angle of rotation.
		
		Returns
		-------
		np.ndarray
			The rotation matrix for the given angle.
		"""
		return np.array([[ np.cos(gamma) ,-np.sin(gamma),             0],
				 [ np.sin(gamma) , np.cos(gamma),             0],
				 [              0,             0,             1]])
	

	@classmethod
	def E_rot(cls, 
		  alpha = None, 
		  beta  = None, 
		  gamma = None) -> np.ndarray:
		"""
		Constructs the rotation matrix for improper euler angles.
		
		Parameters
		----------
		alpha : float
			The angle around which the ``X``-axis is rotated.
		beta : float
			The angle around which the ``Y``-axis is rotated.
		gamma : float
			The angle around which the ``Z``-axis is rotated.
		
		Returns
		-------
		np.ndarray
			The rotation matrix for the improper euler angles.
		"""
		X = cls.X_rot(alpha or 0)
		Y = cls.Y_rot(beta  or 0)
		Z = cls.Z_rot(gamma or 0)
		R = X @ Y @ Z
		return R


	@classmethod
	def Axis_rot(cls, R) -> np.ndarray:
		"""
		Returns
		-------
		np.ndarray
			The axis of rotation for a given rotation matrix or reference frame
		"""
		S = 0.5 * (R - R.T)
		return np.array([S[2, 1], S[0, 2], S[1, 0]])


	@classmethod
	def quat_to_euler(cls, R, I, J, K) -> tuple[float]:
		"""
		Parameters
		----------
		R : float
			The real rotation angle component of the quaternion.
		I : float
			The ``X``-axis component of the quaternion.
		J : float
			The ``Y``-axis component of the quaternion.
		K : float
			The ``Z``-axis component of the quaternion.

		Returns
		-------
		tuple[float]
			The improper euler angles for a given quaternion.
		"""
		R = np.array([[1 - 2*(J**2 + K**2),       2*(I*J - R*K),       2*(R*J + I*K)],
			      [      2*(I*J + R*K), 1 - 2*(I**2 + K**2),       2*(J*K - R*I)],
			      [      2*(I*K - R*J),       2*(R*I + J*K), 1 - 2*(I**2 + J**2)]])
		return cls.reference_frame_to_euler(R)


	@classmethod
	def euler_to_quat(cls, alpha, beta, gamma) -> tuple[float]:
		"""
		Parameters
		----------
		alpha : float
			The angle around which the ``X``-axis is rotated.
		beta : float
			The angle around which the ``Y``-axis is rotated.
		gamma : float
			The angle around which the ``Z``-axis is rotated.

		Returns
		-------
		tuple[float]
			The quaternion components for improper euler angles.
		"""
		R     = cls.E_rot(alpha, beta, gamma)
		axis  = cls.Axis_rot(R)
		angle = np.arccos(0.5 * (R[0, 0] + R[1, 1] + R[2, 2] - 1))
		alpha_prime = cls.angle(axis, cls.X)
		beta_prime  = cls.angle(axis, cls.Y)
		gamma_prime = cls.angle(axis, cls.Z)
		R = np.cos(angle/2)
		I = np.sin(angle/2) * np.cos(alpha_prime)
		J = np.sin(angle/2) * np.cos(beta_prime)
		K = np.sin(angle/2) * np.cos(gamma_prime)
		return R, I, J, K



class Vector:
	"""
	This class implements multiple functions to modify vectors and orientations of MoveableThings.
	"""
	@classmethod
	def get_axis(cls, axis: str|list[int|float]|np.ndarray):
		"""
		This method returns a vector that represents the axis which is either given as a str 
		or as a 1D array like object.
		
		>>> Vector.get_axis("x")
		array([1., 0., 0.])
		>>> Vector.get_axis([0., 1., 0.])
		array([0., 1., 0.])
		>>> Vector.get_axis(np.array([0., 0., 1.]))
		array([0., 0., 1.])
		
		
		Parameters
		----------
		axis : str | list[int | float] | np.ndarray
			The axis argument may be either a 1D array like object or a string ('x', 'y', 'z').
		
		Returns
		-------
		np.ndarray
			The specified axis converted to a numpy array.
		
		Raises
		------
		ValueError
		If the axis argument is a string and neither 'x', 'y' or 'z' a ValueError is raise.
		"""
		if isinstance(axis, str):
			axis = axis.strip().lower()
			if axis not in ('x', 'y', 'z'):
				raise ValueError(f"Attribute axis must be 'x', 'y' or 'z', got '{axis}' instead.")
			if axis == 'x':
				return np.array([1., 0., 0.], dtype=np.float32)
			elif axis == 'y':
				return np.array([0., 1., 0.], dtype=np.float32)
			elif axis == 'z':
				return np.array([0., 0., 1.], dtype=np.float32)
		else:
			return np.array(axis, dtype=np.float32)


	@staticmethod
	def equal(a: np.ndarray, b: np.ndarray, sensitivity: int = 8):
		"""
		Compares whether two vectors are equal within a margin of error as given by sensitivity.
		
		>>> a = np.array([3.2, 0., 0.])
		>>> b = np.array([3.8, 0., 0.])
		>>> c = np.array([3.2, 5., 0.])
		>>> Vector.equal(a, b, sensitivity=0)
		True
		>>> Vector.equal(a, b, sensitivity=1)
		False
		>>> Vector.equal(a, c, sensitivity=0)
		False
		>>> Vector.equal(a, c, sensitivity=-1)
		True
		
		Parameters
		----------
		a : np.ndarray
			A vector to be compared.
		b : np.ndarray
			A vector to be compared.
		sensitivity : int, optional
			The floating point decimal cutoff.
		
		Returns
		-------
		bool
			A boolean indicating whether the two vectors are equal within the specified margin of error.
		"""
		return np.all(np.round(a - b, sensitivity) == 0)


	@classmethod
	def global_position(cls, node, pos=None):
		"""
		This method converts the local position of a Thing (typically a node in a kinematic tree)
		to its global position relative to the nodes root.
		
		Parameters
		----------
		node : Thing
			A Thing (typically a node in a kinematic tree).
		pos : None, optional
			An initial position that is assumed to be located relative to the node.
		
		Returns
		-------
		np.ndarray
			The global position relative to root of the given node, or the initial position relative to the node is pos was specified.
		"""
		pos = pos or np.zeros(3)
		while node is not None:
			if hasattr(node, 'pos'):
				pos = node.pos + node.rotation_matrix @ pos
			node = node.parent
		return pos
	

	@staticmethod
	def normalize(vector: np.ndarray) -> np.ndarray:
		"""
		This method normalizes a vector to a unit lenth of 1.
		
		Parameters
		----------
		vector : np.ndarray
			A vector to be normalized.
		
		Returns
		-------
		np.ndarray
			The normalized vector.
		"""
		norm = np.linalg.norm(vector)
		assert norm > 0, 'The vector norm must be positive for normalization!'
		return vector/norm


	@staticmethod
	def distance(a: np.ndarray|blue.MoveableThingType, b: np.ndarray|blue.MoveableThingType) -> float:
		"""
		This method returns the distance between two vectors or MoveableThings.
		
		Parameters
		----------
		a : np.ndarray | blue.MoveableThingType
			A vector or MoveableThings global position for which the distance is computed w.r.t. b.
		b : np.ndarray | blue.MoveableThingType
			A vector or MoveableThings global position for which the distance is computed w.r.t. a.
		
		Returns
		-------
		float
			The distance betweent the two vectors or MoveableThings.
		"""
		if not isinstance(a, blue.MoveableThingType):
			a = a.global_pos
		if not isinstance(b, blue.MoveableThingType):
			b = b.global_pos
		return np.linalg.norm(b - a)



#class Distance:
#
#	"""
#	This class implements methods to determin the distance between Things.
#	
#	Attributes
#	----------
#	BOX : tuple
#	A collection of Things that are equivalent in form but inherit from different classes (BaseGeom and BaseSite).
#	CAPSULE : tuple
#	A collection of Things that are equivalent in form but inherit from different classes (BaseGeom and BaseSite).
#	CYLINDER : tuple
#	A collection of Things that are equivalent in form but inherit from different classes (BaseGeom and BaseSite).
#	ELLIPSOID : tuple
#	A collection of Things that are equivalent in form but inherit from different classes (BaseGeom and BaseSite).
#	PLANE : tuple
#	A collection of Things that are equivalent in form but inherit from different classes (BaseGeom and BaseSite).
#	SPHERE : tuple
#	A collection of Things that are equivalent in form but inherit from different classes (BaseGeom and BaseSite).
#	TYPES : dict
#	A dictionary of all forms, containing the other collections.
#	"""
#	
#	CAPSULE   = (blue.geoms.Capsule,   blue.sites.Capsule)
#	CYLINDER  = (blue.geoms.Cylinder,  blue.sites.Cylinder)
#	SPHERE    = (blue.geoms.Sphere,    blue.sites.Sphere)
#	ELLIPSOID = (blue.geoms.Ellipsoid, blue.sites.Ellipsoid)
#	BOX       = (blue.geoms.Box,       blue.sites.Box)
#	PLANE     = (blue.geoms.Plane,)
#	TYPES     = ('CAPSULE', 'CYLINDER', 'PLANE', 'SPHERE', 'ELLIPSOID', 'BOX')
#	@classmethod
#	def type(cls, thing):
#		"""
#		Returns the form type of an object for a given Thing.
#		
#		Parameters
#		----------
#		thing : Thing
#			A Thing either geom or site.
#		
#		Returns
#		-------
#		tuple
#			The collection of classes that implement the form of the given Thing.
#		
#		Raises
#		------
#		NotImplemented
#		If the Things type is not implemented, NotImplemented is raised.
#		"""
#		for TYPE in cls.TYPES:
#			if isinstance(thing, cls.__getattribute__(cls, TYPE)):
#				return TYPE
#		raise NotImplemented
#
#
#	def __getattr__(self, attr):
#		"""
#		This method is used to make min distance calls symetric.
#		
#		Parameters
#		----------
#		attr : str
#			A string of the methods name, for example "_min_CAPSULE_CYLINDER"
#		
#		Returns
#		-------
#		method
#			The distance method reflected in arguments.
#		
#		Raises
#		------
#		AttributeError
#		If the attribute does not adhere to the distance function naming conventions an AttributeError is raised.
#		"""
#		if attr.startswith('_min_'):
#			type_a, type_b = attr.replace('_min_', '').split('_')
#			attr           = f'_min_{type_b}_{type_a}'
#			method         = self.__getattribute__(attr)
#			def wrapper(self, thing_a, thing_b):
#				return method(thing_b, thing_a)
#			return wrapper
#		raise AttributeError
#
#
#	def center(thing_a, thing_b):
#		"""
#		A method that returns the distance between the centers of two Things.
#		
#		Parameters
#		----------
#		thing_a : ThingType
#			Description
#		thing_b : ThingType
#			Description
#		
#		Returns
#		-------
#		TYPE
#			Description
#		"""
#		return Vector.distance(thing_a.pos, thing_b.pos)
#
#
#	@classmethod
#	def minimum(cls, thing_a, thing_b):
#		"""
#		Calculates the minimum distance betweeen two Things analytically.
#		Currently the min Distance is approximated with an upper bound, precise 
#		distance calculations will be unrolled over time.
#		
#		Parameters
#		----------
#		thing_a : ThingType
#			A Thing for which the minimum distance is calculated w.r.t. thing_b.
#		thing_b : ThingType
#			A Thing for which the minimum distance is calculated w.r.t. thing_a.
#		
#		Returns
#		-------
#		float
#			The (currently approximated) minimum Distance between the two Things.
#		"""
#		#### FAST HACK
#		pos_a  = Vector.global_position(thing_a)
#		pos_b  = Vector.global_position(thing_b)
#		size_a = np.max(thing_a.size)
#		size_b = np.max(thing_b.size)
#		distance = Vector.distance(thing_a, thing_b)
#		return distance - size_a - size_b
#		#### FUTURE
#		assert isinstance(thing_a, (blue.geoms.Geom, blue.sites.Site)) and isinstance(thing_b, (blue.geoms.Geom, blue.sites.Site))
#		type_a, type_b = cls.type(thing_a), cls.type(thing_b)
#		method_name = f'_min_{type_a}_{type_b}'
#		method = getattr(cls, method_name)
#		return method(thing_a, thing_b)
#
#
#	@classmethod
#	def _min_SPHERE_SPHERE(cls, thing_a, thing_b):
#		"""
#		Calculates the distance between two Spheres analytically
#		
#		Parameters
#		----------
#		thing_a : Sphere
#			A Sphere for which the minimum distance is calculated analytically w.r.t. thing_b.
#		thing_b : Sphere
#			A Sphere for which the minimum distance is calculated analytically w.r.t. thing_a.
#		
#		Returns
#		-------
#		float
#			The minimum distance between two Spheres.
#		"""
#		return cls.center(thing_a, thing_b) - thing_a.radius - thing_b.radius



if __name__ == '__bluen__':
	R = np.stack([Rotation.Y, Rotation.Z, Rotation.X], axis=1)
	alpha, beta, gamma = Rotation.reference_frame_to_euler(R)
	print(alpha, beta, gamma)
	S1 = blue.geoms.Sphere(radius=1.5)
	S2 = blue.geoms.Sphere(pos=[3.5, 0, 0])
	print(Distance.minimum(S1, S2))