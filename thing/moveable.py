import blueprints as blue
from blueprints import restrict
from . import base

import numpy as np
import inspect
import xml.etree.ElementTree as xml
from copy import copy
from collections import defaultdict



class MoveableThing(blue.MoveableThingType, base.BaseThing):
	"""
	This class enables Things to have a position in space and an orientation specified 
	using (improper) euler angles. To place the same Thing in multiple positions and 
	orientations four shortcut methods can be used to construct copies of the Thing on 
	the fly. Those method manipulate position and orientation and this manipulation is 
	either relative to the current position/orientation or absolute, overwriting the current
	position/orientation.


	.. code-block:: python
		:caption: Relative manipulation:

		>>> thing = MoveableThing(pos=[0, 0, 1], alpha=0.8)
		>>> shifted_thing = thing.shift([2, 0, 0])
		>>> shifted_thing.pos
		array([2., 0., 1.], dtype=float32)
		>>> rotated_thing = thing.rotate(beta=0.3)
		>>> rotated_thing.euler
		array([0.8, 0.3, 0. ], dtype=float32)

	.. code-block:: python
		:caption: Absolute manipulation:

		>>> located_thing = thing.locate([2, 0, 0])
		>>> located_thing.pos
		array([2., 0., 0.], dtype=float32)
		>>> oriented_thing = thing.align(beta=0.3)
		>>> oriented_thing.euler
		array([0. , 0.3, 0. ], dtype=float32)

	.. code-block:: python
		:caption: Size manipulation:

		>>> scaled_thing = thing.scaled(3)
		>>> scaled_thing.pos
		array([0., 0., 3.], dtype=float32)
		>>> scaled_thing.size
		array([1.5, 1.5, 1.5], dtype=float32)
	

	For :class:`MoveableThing` instances in a kinematic tree, position and orientation are 
	locally defined, describing the relative change in position and rotation to their parents.
	If one wants to create relocated copies using :meth:`shift` or :meth:`locate` for global 
	coordinates the ``globally`` argument can be set on call.

	Relocation shortcuts also take other :class:`MoveableThing` instances as arguments. This 
	results in a change in position by the arguments position.

	>>> thing_a = MoveableThing(pos=[0, 0, 1])
	>>> thing_b = MoveableThing(pos=[0, 1, 0])
	>>> thing_c = thing_a.shift(pos=thing_b)
	>>> thing_c.pos
	array([0., 1., 1.], dtype=float32)
	"""
	@restrict
	def __init__(self, 
		     pos:   list[int|float]|np.ndarray|None = [0., 0., 0.], 
		     alpha: int|float|None = 0., 
		     beta:  int|float|None = 0., 
		     gamma: int|float|None = 0., 
		     name:  str|None       = None, 
		     x:     int|float|np.int32|np.int64|np.float32|np.float64|None = None, 
		     y:     int|float|np.int32|np.int64|np.float32|np.float64|None = None, 
		     z:     int|float|np.int32|np.int64|np.float32|np.float64|None = None, 
		     quat:  list[int|float]|np.ndarray|None = None, 
		     **kwargs):
		"""
		MoveableThing implements multiple Thing location and orientation in (improper) 
		euler angles in radians.
		
		Parameters
		----------
		pos : list[int | float] | np.ndarray | None, optional
			Represents the position of the object. Changing this attribute also changes the properties :attr:`x`, :attr:`y` and :attr:`z`.
		alpha : int | float | None, optional
			(Improper) euler angle of rotation around the x-axis in radian. Changing this value also changes the :attr:`euler` property.
		beta : int | float | None, optional
			(Improper) euler angle of rotation around the y-axis in radian. Changing this value also changes the :attr:`euler` property.
		gamma : int | float | None, optional
			(Improper) euler angle of rotation around the z-axis in radian. Changing this value also changes the :attr:`euler` property.
		name : str | None, optional
			The user specified name for the Thing.
		x : int | float |np.int32 | np.int64 | np.float32 | np.float64 | None, optional
			If `pos` is not specified, this argument sets the X position coordinate.
		y : int | float |np.int32 | np.int64 | np.float32 | np.float64 | None, optional
			If `pos` is not specified, this argument sets the Y position coordinate.
		z : int | float |np.int32 | np.int64 | np.float32 | np.float64 | None, optional
			If `pos` is not specified, this argument sets the Z position coordinate.
		quat: list [ int | float ] | np.ndarray | None, optional
			If set, the quaternion orientation overwrites the euler angles ``alpha``, ``beta`` and ``gamma``.
		**kwargs
			Keyword arguments are passed to ``super().__init__``.
		"""
		#if pos is not None and any(arg is not None for arg in [x, y, z]):
		#	raise ValueError(f'Only argument (pos) or arguments (x, y, z) can be set. Method call got ({', '.join(name for name, val in [('pos', pos), ('x', x), ('y', y), ('z', z)] if val is not None)}).')
		if pos is not None:
			x = x if x is not None else pos[0]
			y = y if y is not None else pos[1]
			z = z if z is not None else pos[2]
		else:
			x = x if x is not None else 0.
			y = y if y is not None else 0.
			z = z if z is not None else 0.
		pos = np.array([x, y, z], dtype=np.float32)
		#if pos is None:
		#	x = float(x) if x is not None else 0.
		#	y = float(y) if y is not None else 0.
		#	z = float(z) if z is not None else 0.
		#	pos = np.array([x, y, z], dtype=np.float32)
		self.pos    = pos
		if quat is not None:
			R, I, J, K = quat[0], quat[1], quat[2], quat[3]
			alpha, beta, gamma = blue.Rotation.quat_to_euler(R, I, J, K)
		self.alpha  = alpha
		self.beta   = beta
		self.gamma  = gamma
		super().__init__(name=name, **kwargs)


	def view(self) -> None:
		"""
		This method creates a throw away :class:`World` <blueprints.world.World>`, 
		attaches itself to it and uses its viewer.

		Returns
		-------
		None
		"""
		world = blue.World()
		#sky = blue.texture.Skybox(builtin='flat', 
		#	  color_1='white')
		#world.texture = sky
		world.attach(self, copy=True)
		world.view()
		world.unbuild()


	def lattice(self, 
		    directions:  list[float|int]|np.ndarray|list[list[int|float]]|list[np.ndarray], 
		    repetitions: int|list[int]|np.ndarray, 
		    name:        str|None = None) -> None:
		"""
		This method constructs a :class:`Lattice <blueprints.utils.lattice.Lattice>` with the MoveableThing as 
		its components.
		
		Parameters
		----------
		directions : list [ float | int ] | np.ndarray | list [ list [ int | float ] ] | list [ np.ndarray ] 
			A list of axes pointing into the directions of the Lattice.
		repetitions : int | list [ int ] | np.ndarray
			A list of repetitions for each axis with the same length as ``directions``
		name : str | None
			The name of the Lattice (used for displaying purposes).
		

		Returns
		-------
		blue.LatticeType
			A Lattice created with the MoveableThing as components
		"""
		
		return blue.Lattice(thing=self, 
				    directions=directions, 
				    repetitions=repetitions, 
				    name=name)

	# UTILITY METHODS

	@property
	def _location_range(self) -> tuple[np.ndarray, np.ndarray]|None:
		"""
		This function is used to determine the range in which the Thing and its children are located.
		
		Returns
		-------
		tuple[np.ndarray, np.ndarray]
			The returned values are two numpy arrays of shape (3,) containing the 
			minimum and maximum of the coordinate values of each dimension which 
			contain all points of the Thing including its children.
		"""
		if hasattr(self, 'size'):
			pos = blue.geometry.Vector.global_position(self)
			max_size = np.max(np.abs(self.size))
			min_pos = pos - max_size
			max_pos = pos + max_size
			return min_pos, max_pos


	def _update_rotation(self) -> None:
		"""
		This method updates the rotation data in the current mujoco simulation.
		"""
		if isinstance(self.parent, blue.WorldType):
			parent_R = np.eye(3)
		else:
			parent_R = self.parent.global_rotation_matrix
		R = blue.Roration.E_rot(self._alpha, self._beta, self._gamma)
		global_R = parent_R @ R
		data_view = getattr(self.root._mj_data, self._MUJOCO_OBJ)
		data_view(self.name).xmat = global_R.reshape((-1,))


	@restrict
	@staticmethod
	def _get_pos(pos: list[int|float]|np.ndarray|blue.MoveableThingType) -> np.ndarray:
		"""
		This method is used to obtain the position from an input that is either a 
		numeric representation as in a list or a numpy array or a :class:`MoveableThing` 
		from which the position is returned.
		
		Parameters
		----------
		pos : list[int | float] | np.ndarray
			An object either 1D array like object or a :class:`MoveableThing`
		
		Returns
		-------
		np.ndarray
			The position used for further processing.
		"""
		if isinstance(pos, blue.MoveableThingType):
			pos = pos.global_pos
		return np.array(pos, dtype=np.float32)


	@property
	def rotation_matrix(self) -> np.ndarray:
		"""
		The matrix is constructed for local orientation (w.r.t. to parent orientation) — in contrast to 
		global orientation. For this use :meth:`Rotation.global_orientation <blueprints.utils.geometry.Rotation.global_orientation>`.

		Returns
		-------
		np.ndarray
			The rotation matrix for the Thins orientation.
		"""
		if self._launched:
			if isinstance(self.parent, blue.WorldType):
				parent_R = np.eye(3)
			else:
				parent_R = self.parent.global_rotation_matrix
			return parent_R.T @ self.global_rotation_matrix
		else:
			return blue.geometry.Rotation.E_rot(self.alpha, self.beta, self.gamma)
	

	@property
	def global_rotation_matrix(self) -> np.ndarray:
		"""
		The matrix is constructed for global orientation) — in contrast to 
		local orientation. For this use :meth:`Rotation.E_rot <blueprints.utils.geometry.Rotation.Rotation.E_rot>`.

		Returns
		-------
		np.ndarray
			The rotation matrix for the Thins orientation.
		"""
		if self._launched:
			#return self.root._mj_data.__getattibute__(self._MUJOCO_OBJ)(self.name).xmat.reshape((3, 3))
			data_view = getattr(self.root._mj_data, self._MUJOCO_OBJ)
			return data_view(self.name).xmat.reshape((3, 3))
		else:
			return blue.geometry.Rotation.global_rotation_matrix(self)

	# TRANSFORMATION METHODS

	@restrict
	def shift(self, 
		  pos:      list[int|float]|np.ndarray|blue.MoveableThingType|None = None, 
		  globally: bool = False, 
		  x:        int|float|np.int32|np.int64|np.float32|np.float64|None = None, 
		  y:        int|float|np.int32|np.int64|np.float32|np.float64|None = None, 
		  z:        int|float|np.int32|np.int64|np.float32|np.float64|None = None, 
		  **kwargs) -> blue.MoveableThingType:
		"""
		This method creates a copy that is shifted by ``pos``.

		>>> MoveableThing(pos=[5, 0, 0]).shift(x=-3, y=10).pos
		array([2., 10., 0.], dtype=float32)

		Parameters
		----------
		pos : list[int | float] | np.ndarray | blue.MoveableThingType | None
			The amount by which the Thing is shifted, optionally another :class:`MoveableThing` 
			can be passed in which case the position of this Thing is taken.
		globally : bool, optional
			Determining whether the shift is relative to the local orientation or the global orientation.
		x : int | float | np.int32 | np.int64 | np.float32 | np.float64 | None
			This argument can be set if just the X component of the position is to be changed.
		y : int | float | np.int32 | np.int64 | np.float32 | np.float64 | None
			This argument can be set if just the Y component of the position is to be changed.
		z : int | float | np.int32 | np.int64 | np.float32 | np.float64 | None
			This argument can be set if just the Z component of the position is to be changed.
		**kwargs
			Keyword arguments are passed to the :meth:`copy`.
		
		Returns
		-------
		blue.MoveableThingType
			A shifted copy is returned.
		"""
		if pos is not None and any(arg is not None for arg in [x, y, z]):
			raise ValueError(f'Only argument (pos) or arguments (x, y, z) can be set. Method call got ({', '.join(name for name, val in [('pos', pos), ('x', x), ('y', y), ('z', z)] if val is not None)}).')
		if pos is None:
			x = float(x) if x is not None else 0.
			y = float(y) if y is not None else 0.
			z = float(z) if z is not None else 0.
			pos = np.array([x, y, z], dtype=np.float32)
		pos   = self._get_pos(pos)
		thing = self.copy(**kwargs)
		if globally:
			rotation_matrix = blue.geometry.Rotation.global_rotation_matrix(self)
			pos = rotation_matrix.T @ pos
		thing.pos = thing.pos + pos
		return thing


	@restrict
	def locate(self, 
		   pos:      list[int|float]|np.ndarray|blue.MoveableThingType|None = None, 
		   globally: bool = False, 
		   x:        int|float|np.int32|np.int64|np.float32|np.float64|None = None, 
		   y:        int|float|np.int32|np.int64|np.float32|np.float64|None = None, 
		   z:        int|float|np.int32|np.int64|np.float32|np.float64|None = None, 
		   **kwargs) -> blue.MoveableThingType:
		"""
		This method creates a copy that is relocated to ``pos``.
		
		>>> MoveableThing(pos=[5, 0, 0]).locate(x=-3, y=10).pos
		array([-3., 10., 0.], dtype=float32)

		Parameters
		----------
		pos : list[int | float] | np.ndarray | blue.MoveableThingType | None
			The position to which the Thing is shifted, optionally another :class:`MoveableThing` 
			can be passed in which case the position of this Thing is taken.
		globally : bool, optional
			Determining whether the relocation is relative to the local orientation or the global orientation.
		x : int | float | np.int32 | np.int64 | np.float32 | np.float64 | None
			This argument can be set if just the ``X`` component of the position is to be changed.
		y : int | float | np.int32 | np.int64 | np.float32 | np.float64 | None
			This argument can be set if just the ``Y`` component of the position is to be changed.
		z : int | float | np.int32 | np.int64 | np.float32 | np.float64 | None
			This argument can be set if just the ``Z`` component of the position is to be changed.
		**kwargs
			Keyword arguments are passed to :meth:`copy`.
		
		Returns
		-------
		blue.MoveableThingType
			A relocated copy is returned.
		"""
		if pos is not None and sum(arg is not None for arg in [x, y, z]) > 0:
			raise ValueError(f'Only argument (pos) or arguments (x, y, z) can be set. Method call got ({', '.join(name for name, val in [('pos', pos), ('x', x), ('y', y), ('z', z)] if val is not None)}).')
		if pos is None:
			x = float(x) if x is not None else 0.
			y = float(y) if y is not None else 0.
			z = float(z) if z is not None else 0.
			pos = np.array([x, y, z], dtype=np.float32)
		pos   = self._get_pos(pos)
		thing = self.copy(**kwargs)
		if globally:
			rotation_matrix = blue.geometry.Rotation.global_rotation_matrix(self)
			pos = rotation_matrix.T @ pos
		thing.pos = np.array(pos)
		return thing


	@restrict
	def rotate(self, 
		   alpha:    int|float = 0, 
		   beta:     int|float = 0, 
		   gamma:    int|float = 0, 
		   quat:     list[int|float]|np.ndarray|None = None, 
		   globally: bool      = True, 
		   center:   list[int|float]|np.ndarray|blue.MoveableThingType|None = None, 
		   **kwargs) -> blue.MoveableThingType:
		"""
		This method creates a copy that is rotated by the (improper) euler angles.

		>>> MoveableThing(alpha=TAU/4, beta=0, gamma=TAU/8).rotate(beta=PI).euler
		array([-TAU/4,  PI,  TAU/8], dtype=float32)

		Parameters
		----------
		alpha : int | float, optional
			The (improer) euler angle of rotation around the ``X``-axis in radians.
		beta : int | float, optional
			The (improer) euler angle of rotation around the ``Y``-axis in radians.
		gamma : int | float, optional
			The (improer) euler angle of rotation around the ``Z``-axis in radians.
		globally : bool, optional
			Determining whether the rotation is relative to the local orientation or the global orientation.
		center : list[int | float] | np.ndarray | blue.MoveableThingType | None, optional
			The copied Thing will be rotated around center. If center is None, it will be rotated at its position instead.
		**kwargs
			Keyword arguments are passed to the :meth:`copy`.
		
		Returns
		-------
		blue.MoveableThingType
			A copy of the returned Thing is returned.
		"""
		thing = self.copy(**kwargs)
		if quat is not None:
			R, I, J, K = quat[0], quat[1], quat[2], quat[3]
			alpha, beta, gamma = blue.Rotation.quat_to_euler(R, I, J, K)
		if globally:
			R1 = blue.geometry.Rotation.global_rotation_matrix(self)
			R2 = blue.geometry.Rotation.E_rot(alpha, beta, gamma)
			R  = R2 @ R1
		else:
			R1 = thing.rotation_matrix
			R2 = blue.geometry.Rotation.E_rot(alpha, beta, gamma)
			R  = R1 @ R2
		alpha, beta, gamma = blue.geometry.Rotation.reference_frame_to_euler(R)
		thing.alpha = alpha
		thing.beta  = beta
		thing.gamma = gamma
		if center is not None:
			center = self._get_pos(center)
			if globally:
				thing.global_pos = R2 @ (thing.global_pos - center) + center
			else:
				thing.pos = R2 @ (thing.pos - center) + center
		return thing


	@restrict
	def align(self, 
		  alpha:    int|float = 0., 
		  beta:     int|float = 0., 
		  gamma:    int|float = 0., 
		  quat:     list[int|float]|np.ndarray|None = None, 
		   **kwargs) -> blue.MoveableThingType:
		"""
		This method creates a copy that is oriented according to the (improper) euler angles.
		
		>>> MoveableThing(alpha=TAU/4, beta=0, gamma=TAU/8).align(beta=PI).euler
		array([0,  PI,  0], dtype=float32)

		Parameters
		----------
		alpha : int | float, optional
			The (improer) euler angle of rotation around the ``X``-axis in radians.
		beta : int | float, optional
			The (improer) euler angle of rotation around the ``Y``-axis in radians.
		gamma : int | float, optional
			The (improer) euler angle of rotation around the ``Z``-axis in radians.
		**kwargs
			Keyword arguments are passed to the :meth:`copy`.
		
		Returns
		-------
		blue.MoveableThingType
			A copy of the Thing set to the given orientation is returned.
		"""
		thing = self.copy(**kwargs)
		if quat is not None:
			R, I, J, K = quat[0], quat[1], quat[2], quat[3]
			alpha, beta, gamma = blue.Rotation.quat_to_euler(R, I, J, K)
		thing.alpha = alpha if alpha is not None else 0
		thing.beta  = beta  if beta  is not None else 0
		thing.gamma = gamma if gamma is not None else 0
		return thing


	@restrict
	def scaled(self, scale: int|float, keep_pos: bool = False, **kwargs) -> blue.MoveableThingType:
		"""
		This method creates a copy that is rescaled in size and position.

		scale : int | float | None, optional
			The scaling factor by which the Things size and position is increased.
		keep_pos : bool
			If set, the position is kept, otherwise, the position is scaled as well.
		**kwargs
			Keyword arguments are passed to the :meth:`copy`.
		
		Returns
		-------
		blue.MoveableThingType
			A copy of the rescaled Thing.
		"""
		if isinstance(self, blue.NodeThingType):
			children  = {name: [thing.scaled(scale, keep_pos) for thing in list(child_dict['children'])] \
									 for name, child_dict in self._CHILDREN.items() \
									 if name not in kwargs and not issubclass(child_dict['type'], blue.CyclicalThingType)}
			cyclicals = {name: [thing.scaled(scale, keep_pos) for thing in list(child_dict['children'])] \
									 for name, child_dict in self._CHILDREN.items() \
									 if name not in kwargs and     issubclass(child_dict['type'], blue.CyclicalThingType)}
			blueprint_specs = self._blueprint_specs()
			blueprint_specs.update(children)
			blueprint_specs.update(kwargs)
			if 'name' not in kwargs:
				blueprint_specs['name'] = self._name
			thing = self.__class__(copy=False, **blueprint_specs)
			self._migrate_children(thing, cyclicals)
		else:
			thing     = self.copy(**kwargs)
		if not keep_pos:
			thing.pos = thing.pos * scale
		if hasattr(thing, 'size'):
			thing.size = thing.size * scale
		return thing

	# MUJOCO PROPERTIES

	@property
	def euler(self) -> np.ndarray:
		"""
		Note that euler angles are not additive for rotations. Use :meth:`rotate` for this.
		
		Returns
		-------
		np.ndarray
			The orientation is represented as a vector containing the (improper) euler 
			angles array([alpha, beta, gamma]).
		"""
		if self._launched:
			R = self.rotation_matrix
			alpha, beta, gamma = blue.Rotation.reference_frame_to_euler(R)
			return np.array([alpha, 
					 beta,
					 gamma], dtype=np.float32)
		else:
			return np.array([self.alpha, 
					 self.beta,
					 self.gamma], dtype=np.float32)


	@euler.setter
	@restrict
	def euler(self, value: list[int|float]|np.ndarray) -> None:
		"""
		Note that euler angles are not additive for rotations. Use :meth:`rotate` for this.
		
		Parameters
		----------
		value : list[int | float] | np.ndarray
			The orientation is represented as a vector containing the (improper) euler 
			angles array([alpha, beta, gamma]).

		Raises
		------
		ValueError
			If the (improper) euler angles do not have the correct shape an error is raised.
		"""
		if isinstance(value, np.ndarray) and value.shape != (3,):
			raise ValueError(f'Orientation attribute euler must have a dimension of 3 got {value.shape} instead.')
		if isinstance(value, list) and len(value) != 3:
			raise ValueError(f'Orientation attribute euler must have a dimension of 3 got {len(value)} instead.')
		self._alpha, self._beta, self._gamma = map(float, value)
		if self._launched:
			self._update_rotation()


	@property
	def pos(self) -> np.ndarray:
		"""
		Individual components of :attr:`pos` can be found in :attr:`x`, :attr:`y` and :attr:`z`.

		Returns
		-------
		np.ndarray
			The position of the Thing is given in local coordinates.
		"""
		if self._launched:
			if isinstance(self.parent, blue.WorldType):
				parent_pos = np.zeros(3)
				parent_R   = np.eye(3)
			else:
				parent_pos = self.parent.global_pos
				parent_R   = self.parent.global_rotation_matrix
			delta_pos = self.global_pos - parent_pos
			return parent_R.T @ delta_pos
		else:
			return self._pos.copy()


	@pos.setter
	@restrict
	def pos(self, pos: np.ndarray|list[int|float]) -> None:
		"""
		Individual components of :attr:`pos` can be found in :attr:`x`, :attr:`y` and :attr:`z`.

		Parameters
		----------
		pos : np.ndarray | list[int | float]
			The position is always set in local coordinates.
		
		Raises
		------
		ValueError
			If the position does not have the correct shape an error is raised.
		"""
		if isinstance(pos, np.ndarray) and pos.shape != (3,):
			raise ValueError(f'Position attribute pos must have a dimension of 3 got {pos.shape} instead.')
		if isinstance(pos, list) and len(pos) != 3:
			raise ValueError(f'Position attribute pos must have a dimension of 3 got {len(pos)} instead.')
		pos = np.array(pos, dtype=np.float32)
		if self._launched:
			array = self.root._mj_model.__getattribute__(f'{self._MUJOCO_DATA}_pos')
			array[self._index] = pos
			if isinstance(self, blue.BodyType):
				delta_pos  = pos - self.pos
				global_pos = self.global_pos
				if isinstance(self.parent, blue.WorldType):
					R = np.eye(3)
				else:
					R = self.parent.global_rotation_matrix
				self.root._mj_data.xpos[self._index] = global_pos + R @ delta_pos
		self._pos = pos


	@property
	def global_pos(self):
		"""
		Individual components of :attr:`global_pos` can be found in :attr:`global_x`, :attr:`global_y` and :attr:`global_z`.

		Returns
		-------
		np.ndarray
			The position of the Thing is given in global coordinates.
		"""
		if self._launched:
			pos = np.zeros(3)
			for node in self.path[::-1]:
				if isinstance(node, blue.BodyType):
					global_R   = blue.Rotation.global_rotation_matrix(node)
					global_pos = self.root._mj_data.xpos[node._index]
					pos = global_pos + global_R @ pos
					break
				else:
					pos = node._pos + node.rotation_matrix @ pos
			return pos
		else:
			return blue.Vector.global_position(self)


	@global_pos.setter
	@restrict
	def global_pos(self, pos: np.ndarray|list[int|float]) -> None:
		"""
		Individual components of :attr:`pos` can be found in :attr:`x`, :attr:`y` and :attr:`z`.

		Parameters
		----------
		pos : np.ndarray | list[int | float]
			The position is always set in local coordinates.
		
		Raises
		------
		ValueError
			If the position does not have the correct shape an error is raised.
		"""
		if isinstance(pos, np.ndarray) and pos.shape != (3,):
			raise ValueError(f'Position attribute pos must have a dimension of 3 got {pos.shape} instead.')
		if isinstance(pos, list) and len(pos) != 3:
			raise ValueError(f'Position attribute pos must have a dimension of 3 got {len(pos)} instead.')
		pos = np.array(pos, dtype=np.float32)
		delta_pos = pos - self.global_pos
		if isinstance(self.parent, blue.WorldType) or self.parent is None:
			global_R = np.eye(3)
		else:
			global_R  = self.parent.global_rotation_matrix
		if self._launched:
			array = self.root._mj_model.__getattribute__(f'{self._MUJOCO_DATA}_pos')
			array[self._index] += global_R.T @ delta_pos
			if isinstance(self, blue.BodyType):
				self.root._mj_data.xpos[self._index] = pos
		self._pos += global_R.T @ delta_pos


	@property
	def alpha(self) -> float|None:
		"""
		Note that euler angles are not additive for rotations. Use :meth:`rotate` for this.
		
		Returns
		-------
		float | None
			The (improper) euler angle of rotation around the X-axis.
		"""
		if self._launched:
			return float(self.euler[0])
		else:
			return self._alpha


	@alpha.setter
	@restrict
	def alpha(self, alpha: int|float|np.int32|np.int64|np.float32|np.float64) -> None:
		"""
		Note that euler angles are not additive for rotations. Use :meth:`rotate` for this.
		
		Parameters
		----------
		alpha : int | float | np.int32 | np.int64 | np.float32 | np.float64
			The (improper) euler angle of rotation around the X-axis.
		"""
		self._alpha = float(alpha)
		if self._launched:
			self._update_rotation()


	@property
	def beta(self) -> float|None:
		"""
		Note that euler angles are not additive for rotations. Use :meth:`rotate` for this.
		
		Returns
		-------
		float | None
			The (improper) euler angle of rotation around the Y-axis.
		"""
		if self._launched:
			return float(self.euler[1])
		else:
			return self._beta


	@beta.setter
	@restrict
	def beta(self, beta: int|float|np.int32|np.int64|np.float32|np.float64) -> None:
		"""
		Note that euler angles are not additive for rotations. Use :meth:`rotate` for this.
		
		Parameters
		----------
		beta : int | float | np.int32 | np.int64 | np.float32 | np.float64
			The (improper) euler angle of rotation around the Y-axis.
		"""
		self._beta = float(beta)
		if self._launched:
			self._update_rotation()


	@property
	def gamma(self) -> float|None:
		"""
		Note that euler angles are not additive for rotations. Use :meth:`rotate` for this.
		
		Returns
		-------
		float | None
			The (improper) euler angle of rotation around the Z-axis.
		"""
		if self._launched:
			return float(self.euler[2])
		else:
			return self._gamma


	@gamma.setter
	@restrict
	def gamma(self, gamma: int|float|np.int32|np.int64|np.float32|np.float64) -> None:
		"""
		Note that euler angles are not additive for rotations. Use :meth:`rotate` for this.
		
		Parameters
		----------
		gamma : int | float | np.int32 | np.int64 | np.float32 | np.float64
			The (improper) euler angle of rotation around the Z-axis.
		"""
		self._gamma = float(gamma)
		if self._launched:
			self._update_rotation()


	@property
	def quat(self):
		"""
		Alternatively `quaternions <https://en.wikipedia.org/wiki/Quaternion>`__ can be used for orientations. 
		Where every euler angles are provided, a ``quat`` can be passed instead.

		Returns
		-------
		np.ndarray
		"""
		quat = blue.Rotation.euler_to_quat(self.alpha. self.beta, self.gamma)
		return np.array(quat, dtype=np.float32)


	@quat.setter
	@blue.restrict
	def quat(self, quat: list[int|float]|np.ndarray):
		R, I, J, K = quat[0], quat[1], quat[2], quat[3]
		alpha, beta, gamma = blue.Rotation.quat_to_euler(R, I, J, K)
		self.alpha = alpha
		self.beta  = beta
		self.gamma = gamma
		
	


	@property
	def x(self) -> float|None:
		"""
		If the whole position should be updated use :attr:`pos` instead.
		
		Returns
		-------
		float | None
			The ``X``-coordinate of the Things local position.
		"""
		return float(self.pos[0])


	@x.setter
	@restrict
	def x(self, x: int|float|np.int32|np.int64|np.float32|np.float64) -> None:
		"""
		If the whole position should be updated use :attr:`pos` instead.
		
		Parameters
		----------
		x : int | float | np.int32 | np.int64 | np.float32 | np.float64
			The X-coordinate of the Things local position.
		"""
		pos      = self.pos
		pos[0]   = float(x)
		self.pos = pos


	@property
	def y(self) -> float|None:
		"""
		If the whole position should be updated use :attr:`pos` instead.
		
		Returns
		-------
		float | None
			The ``Y``-coordinate of the Things local position.
		"""
		return float(self.pos[1])


	@y.setter
	@restrict
	def y(self, y: int|float|np.int32|np.int64|np.float32|np.float64) -> None:
		"""
		If the whole position should be updated use :attr:`pos` instead.
		
		Parameters
		----------
		y : int | float | np.int32 | np.int64 | np.float32 | np.float64
			The Y-coordinate of the Things local position.
		"""
		pos      = self.pos
		pos[1]   = float(y)
		self.pos = pos


	@property
	def z(self) -> float|None:
		"""
		If the whole position should be updated use :attr:`pos` instead.
		
		Returns
		-------
		float | None
			The ``Z``-coordinate of the Things local position.
		"""
		return float(self.pos[2])


	@z.setter
	@restrict
	def z(self, z: int|float|np.int32|np.int64|np.float32|np.float64) -> None:
		"""
		If the whole position should be updated use :attr:`pos` instead.
		
		Parameters
		----------
		z : int | float | np.int32 | np.int64 | np.float32 | np.float64
			The Z-coordinate of the Things local position.
		"""
		pos      = self.pos
		pos[2]   = float(z)
		self.pos = pos


	@property
	def vel(self) -> np.ndarray:
		"""
		The global velocity for all axis
		
		Returns
		-------
		np.ndarray
		"""
		if self._launched:
			node = self
			while not isinstance(node, blue.BodyType):
				node = node.parent
				if node is None:
					return np.zeros(3)
			vel = self.root._mj_data.body(node._index).cvel[3:]
			return np.array(vel)
		else:
			return np.zeros(3)


	@property
	def x_vel(self):
		"""
		The global velocity of the ``X`` axis
		
		Returns
		-------
		np.ndarray
		"""
		return self.vel[0]


	@property
	def y_vel(self):
		"""
		The global velocity of the ``Y`` axis
		
		Returns
		-------
		np.ndarray
		"""
		return self.vel[1]


	@property
	def z_vel(self):
		"""
		The global velocity of the ``Z`` axis
		
		Returns
		-------
		np.ndarray
		"""
		return self.vel[2]
	


	# NO EFFECT ON SIMULATION
	#@vel.setter
	#@restrict
	#def vel(self, vel: np.ndarray|list[int|float]) -> None:
	#	"""
	#	Parameters
	#	-------
	#	vel : np.ndarray | list[int | float]
	#		The global velocity for all axis
	#	"""
	#	if isinstance(vel, np.ndarray) and vel.shape != (3,):
	#		raise ValueError(f'Argument vel must be of shape (3,) but received was {vel.shape}!')
	#	elif len(vel) != 3:
	#		raise ValueError(f'Argument vel must be of shape (3,) but received was ({len(vel)},)!')
	#	if self._launched:
	#		node = self
	#		while not isinstance(node, blue.BodyType):
	#			node = node.parent
	#			if node is None:
	#				return
	#		self.root._mj_data.body(node._index).cvel[3:] = vel


	@property
	def angular_vel(self) -> np.ndarray:
		"""
		The global angular velocity for all axis
		
		Returns
		-------
		np.ndarray
		"""
		if self._launched:
			node = self
			while not isinstance(node, blue.BodyType):
				node = node.parent
				if node is None:
					return np.zeros(3)
			vel = self.root._mj_data.body(node._index).cvel[:3]
			return np.array(vel)
		else:
			return np.zeros(3)


	# NO EFFECT ON SIMULATION
	#@angular_vel.setter
	#@restrict
	#def angular_vel(self, angular_vel: np.ndarray|list[int|float]) -> None:
	#	"""
	#	Parameters
	#	-------
	#	angular_vel : np.ndarray | list[int | float]
	#		The global angular velocity for all axis
	#	"""
	#	if isinstance(angular_vel, np.ndarray) and angular_vel.shape != (3,):
	#		raise ValueError(f'Argument angular_vel must be of shape (3,) but received was {angular_vel.shape}!')
	#	elif len(angular_vel) != 3:
	#		raise ValueError(f'Argument angular_vel must be of shape (3,) but received was ({len(angular_vel)},)!')
	#	if self._launched:
	#		node = self
	#		while not isinstance(node, blue.BodyType):
	#			node = node.parent
	#			if node is None:
	#				return
	#		self.root._mj_data.body(node._index).cvel[:3] = angular_vel
