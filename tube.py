import numpy as np
import blueprints as blue



class BaseTube(blue.TubeType):

	"""
	BaseTube is a base class for sites and geoms that can be defined by two points forming a tube like object.
	"""
	
	@blue.restrict
	@classmethod
	def from_points(cls, 
			head:   np.ndarray|list[int|float], 
			tail:   np.ndarray|list[int|float], 
			radius: int|float = 1, 
			**kwargs) -> blue.TubeType:
		"""
		Constructs a Tube like object from point head to point tail.
		Euler angles are set such that the resulting orientation
		has pitch and yaw but no roll. Therefore the floors horizon
		is always parallel to the orientation.
		
		Args:
			head (np.ndarray | list[int | float]): Description
			tail (np.ndarray | list[int | float]): Description
			radius (int | float, optional): Description
			**kwargs: Description
		
		Returns:
			blue.Thing: Description
		"""
		head   = np.array(head)
		tail   = np.array(tail)
		path   =  tail - head
		pos    = (head + tail)/2
		length = np.linalg.norm(path)
		# RECONSTRUCT REFERENCE FRAME
		R_z    = path/length
		R_z1, R_z2, R_z3 = R_z
		if R_z1 == R_z2 == 0:
			alpha = 0 if R_z3 > 0 else blue.geometry.PI
			beta  = 0
			gamma = 0
		else:
			R_x = blue.geometry.Vector.normalize(np.array([R_z2, -R_z1, 0]))
			R_y = np.cross(R_z, R_x)
			R   = np.stack([R_x, R_y, R_z], axis=1)
			# RECONSTRUCT ANGLES
			alpha, beta, gamma = blue.geometry.Rotation.reference_frame_to_euler(R)
		kwargs['pos']    = pos
		kwargs['alpha']  = alpha
		kwargs['beta']   = beta
		kwargs['gamma']  = gamma
		if issubclass(cls, (blue.BoxGeomType, blue.BoxSiteType)):
			kwargs['x_length'] = radius
			kwargs['y_length'] = radius
			kwargs['z_length'] = length
		else:
			kwargs['length'] = length
			kwargs['radius'] = radius
		return cls(**kwargs)


	@property
	def head(self) -> np.ndarray:
		"""
		Returns the head position of the tube like object. The positions are derived even if the object was not constructed with the from points method.
		
		Returns:
			np.ndarray: The head position.
		"""
		R = self.rotation_matrix
		length = self.z_length if isinstance(self, (blue.BoxGeomType, blue.BoxSiteType)) else self.length
		Z = blue.geometry.Rotation.Z * length / 2
		return self.pos - R @ Z


	@property
	def tail(self) -> np.ndarray:
		"""
		Returns the tail position of the tube like object. The positions are derived even if the object was not constructed with the from points method.
		
		Returns:
			np.ndarray: The tail position.
		"""
		R = self.rotation_matrix
		length = self.z_length if isinstance(self, (blue.BoxGeomType, blue.BoxSiteType)) else self.length
		Z = blue.geometry.Rotation.Z * length / 2
		return self.pos + R @ Z


	@head.setter
	@blue.restrict
	def head(self, head: np.ndarray|list[int|float]) -> None:
		"""
		Setter for the head parameter. The position and orientation of the object is adjusted accordingly.
		
		Args:
			head (np.ndarray | list[int | float]): Description
		
		Returns:
			blue.Thing: The new Geom or Site with the adjusted position and orientation.
		"""
		tube        = self.from_points(head, self.tail, **self._blueprint_specs())
		self.pos    = tube.pos
		self.euler  = tube.euler
		self.length = tube.length


	@tail.setter
	@blue.restrict
	def tail(self, tail: np.ndarray|list[int|float]) -> None:
		"""
		Setter for the tail parameter. The position and orientation of the object is adjusted accordingly.
		
		Args:
			tail (np.ndarray | list[int | float]): Description
		
		Returns:
			blue.Thing: The new Geom or Site with the adjusted position and orientation.
		"""
		tube        = self.from_points(self.head, tail, **self._blueprint_specs())
		self.pos    = tube.pos
		self.euler  = tube.euler
		self.length = tube.length
