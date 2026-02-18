import os
import sys
import numpy as np
import xml.etree.ElementTree as xml
import blueprints as blue



class BaseGeom(blue.GeomType, blue.thing.NodeThing, blue.thing.MoveableThing, blue.thing.ColoredThing):

	"""
	Geoms introduce physical matter into the simulation. For different Geom shapes individual
	Geom classes are available, with shape depended size attributes. All other attributes are 
	specified in :class:`BaseGeom`. 

	.. note::
		Though units are not bound to one specific system of measurements, it is highly recommended 
		to use `MKS <https://en.wikipedia.org/wiki/MKS_system_of_units>`__, since all defaults are 
		defined based on meters, kilograms and seconds and most mujoco models online also use the 
		metric system. If you want to switch from the imperial to the metric system the following resource might be 
		of interest to `convert units from imperial to metric <https://xkcd.com/526/>`__. If you insist on 
		using imperial units, take a look at the following resource to `convert units from metric to imperial <https://rick.nerial.uk/>`__.
	
	Most attribute descriptions are partially taken from `Mujoco <https://mujoco.readthedocs.io/en/latest/XMLreference.html#body-geom>`__.
	"""
	
	@blue.restrict
	def __init__(self, 
		     pos:                np.ndarray|list[int|float] = [0., 0., 0.],
		     alpha:              int|float = 0,
		     beta:               int|float = 0,
		     gamma:              int|float = 0, 
		     material:           blue.MaterialType|None = None, 
		     mass:               int|float|None = None,
		     density:            int|float      = 1000.,
		     margin:             int|float      = 0.0,
		     gap:                int|float      = 0.0,
		     sliding_friction:   int|float      = 1,
		     torsional_friction: int|float      = 0.005,
		     rolling_friction:   int|float      = 0.0001, 
		     shellinertia:       bool|None      = None,
		     color:              object|None    = None, 
		     name:               str|None       = None, 
		     x:                  int|float|np.int32|np.int64|np.float32|np.float64|None = None, 
		     y:                  int|float|np.int32|np.int64|np.float32|np.float64|None = None, 
		     z:                  int|float|np.int32|np.int64|np.float32|np.float64|None = None, 
		     quat:               list[int|float]|np.ndarray|None = None, 
		     **kwargs):
		"""
		Parameters
		----------
		pos : list[int | float] | np.ndarray | None, optional
			Represents the position of the object. Changing this attribute also changes the properties :attr:`x`, :attr:`y` and :attr:`z`.
		alpha : int | floatobject | None, optional
			(Improper) euler angle of rotation around the x-axis in radian. Changing this value also changes the :attr:`euler` property.
		beta : int | float | None, optional
			(Improper) euler angle of rotation around the y-axis in radian. Changing this value also changes the :attr:`euler` property.
		gamma : int | float | None, optional
			(Improper) euler angle of rotation around the z-axis in radian. Changing this value also changes the :attr:`euler` property.
		material : blue.MaterialType, optional
			The :class:`Material <blueprints.material.Material>` of the Geom. 
			:class:`Textures <blueprints.texture.BaseTexture> are applied via Materials.
		density : int | float | None, optional
			Material density used to compute the geom mass and inertia. The computation is based on 
			the geom shape and the assumption of uniform density. The internal default of 1000 is the 
			density of water in SI units. This attribute is used only when the mass attribute above is 
			unspecified.
		margin : int | float | None, optional
			A contact is considered active only if the distance between the two geom surfaces is below 
			margin-gap.
		gap : int | float | None, optional
			This attribute is used to enable the generation of inactive contacts, i.e., contacts that are 
			ignored by the constraint solver but are included in mjData.contact for the purpose of custom 
			computations. When this value is positive, geom distances between margin and margin-gap 
			correspond to such inactive contacts.
		mass : int | float | None, optional
			If this attribute is specified, the density attribute below is ignored and the geom density
			is computed from the given mass, using the geom shape and the assumption of uniform density. 
			The computed density is then used to obtain the geom inertia. Recall that the geom mass and 
			inertia are only used during compilation, to infer the body mass and inertia if necessary.
		shellinertia : bool | None, optional
			If true, the geom’s inertia is computed assuming that all the mass is concentrated on the 
			boundary. In this case density is interpreted as surface density rather than volumetric 
			density.
		sliding_friction : int | float, optional
			Friction parameter for rolling used to computed the forces on contact pairs. 
		torsional_friction : int | float, optional
			Friction parameter for sliding used to computed the forces on contact pairs. 
		rolling_friction : int | float, optional
			Friction parameter for torsion used to computed the forces on contact pairs. 
		color : blue.ColorType
			The color of the Geom. For a detailed description see :class:`Color `
		name : str | None, optional
			The user specified name might potentially be altered to avoid a naming conflict by appending an enumeration scheme.
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
		self.mass               = mass # float(mass) if mass is not None else mass
		self.density            = density
		self.shellinertia       = shellinertia
		self.margin             = margin
		self.gap                = gap
		self.sliding_friction   = sliding_friction
		self.torsional_friction = torsional_friction
		self.rolling_friction   = rolling_friction
		self.margin             = margin
		# MATERIAL
		self.material           = material
		# PSEUDO CHILDREN
		self._tendons           = []
		self._CHILDREN 		= {'tendons': {'type':     blue.TendonType, 
						       'children': self._tendons}}
		# SETTING COLOR BLUEPRINT COLOR DEFAULT FOR GEOMS
		color = color if color is not None else 'grey'
		super().__init__(pos=pos, 
				 alpha=alpha, 
				 beta=beta, 
				 gamma=gamma, 
				 name=name, 
				 color=color, 
				 x=x, 
				 y=y, 
				 z=z, 
				 quat=quat, 
				 **kwargs)


	@blue.restrict
	def _build(self, parent, world, indicies, **kwargs):
		"""
		This method is called to build the xml.
		
		Parameters
		----------
		parent : xml.etree.ElementTree.Element
			The xml element of its parent
		world : WorldType
			The World from which the build method was called initially
		indicies : dict
			All indecies used for retrieving data during the simulation.
		
		Returns
		-------
		xml.etree.ElementTree.Element
			The builded xml element of the Thing.
		"""
		self._index       = indicies['geom']
		indicies['geom'] += 1
		if self.material is not None:
			self.material._build(parent, world, indicies, **kwargs)
			kwargs['material'] = self.material.asset.name
		return super()._build(parent, world, indicies, **kwargs)


	@property
	def type(self) -> str:
		"""
		This is a derived attribute, which is used to specify the ``type`` attribute in the 
		mujoco ``geom`` tag.

		Returns
		-------
		str
			The type is the lower case name of the Geom class.
		"""
		return self.__class__.__name__.lower()


	@property
	def material(self):
		"""
		Materials can be used to specify reflective properties of a Geom, 
		as well as to apply :class:`Textures <blueprints.texture.BaseTexture>` to them.

		Returns
		-------
		blue.MaterialType
			The Material of the Geom
		"""
		return self._material


	@material.setter
	@blue.restrict
	def material(self, material: blue.MaterialType|None) -> None:
		self._material = material.copy() if material is not None else material


	@property
	def friction(self) -> np.ndarray:
		"""
		Contact friction parameters for dynamically generated contact pairs. The first number is 
		the sliding friction, acting along both axes of the tangent plane. The second number is the 
		torsional friction, acting around the contact normal. The third number is the rolling 
		friction, acting around both axes of the tangent plane.

		Returns
		-------
		np.ndarray
			Individual components are found in :attr:`sliding_friction`, :attr:`torsional_friction` 
			and :attr:`rolling_friction`.
		"""
		return np.array([self.sliding_friction, 
				 self.torsional_friction, 
				 self.rolling_friction], dtype=np.float32)


	@friction.setter
	@blue.restrict
	def friction(self, friction: np.ndarray|list[int|float]):
		"""
		Contact friction parameters for dynamically generated contact pairs. The first number is 
		the sliding friction, acting along both axes of the tangent plane. The second number is the 
		torsional friction, acting around the contact normal. The third number is the rolling 
		friction, acting around both axes of the tangent plane.

		Parameters
		----------
		friction : np.ndarray | list[int | float]
			Individual components are found in :attr:`sliding_friction`, :attr:`torsional_friction` 
			and :attr:`rolling_friction`.
		"""
		self.sliding_friction   = friction[0]
		self.torsional_friction = friction[1]
		self.rolling_friction   = friction[2]


	@property
	def mass(self) -> float:
		"""
		If this attribute is specified, the density attribute below is ignored and the geom density
		is computed from the given mass, using the geom shape and the assumption of uniform density. 
		The computed density is then used to obtain the geom inertia. Recall that the geom mass and 
		inertia are only used during compilation, to infer the body mass and inertia if necessary.

		Unit defaults are set in SI units, but the physical properties can in principle be 
		interpreted in any system of measurements.

		Returns
		-------
		float
			The default unit convention is kilogram.
		"""
		return self._mass


	@mass.setter
	@blue.restrict
	def mass(self, mass: int|float|None) -> None:
		"""
		If this attribute is specified, the density attribute below is ignored and the geom density
		is computed from the given mass, using the geom shape and the assumption of uniform density. 
		The computed density is then used to obtain the geom inertia. Recall that the geom mass and 
		inertia are only used during compilation, to infer the body mass and inertia if necessary.

		Unit defaults are set in SI units, but the physical properties can in principle be 
		interpreted in any system of measurements.

		Parameters
		----------
		mass : int | float | None
			The default unit convention is kilogram.
		"""
		self._mass = float(mass) if mass is not None else self._DEFAULT_VALS()['mass']


	@property
	def density(self) -> float:
		"""
		Unit defaults are set in SI units, but the physical properties can in principle be 
		interpreted in any system of measurements.

		Material density used to compute the geom mass and inertia. The computation is based on 
		the geom shape and the assumption of uniform density. The internal default of 1000 is the 
		density of water in SI units. This attribute is used only when the mass attribute above is 
		unspecified.
		
		Returns
		-------
		float
			Density is either volumetric if :attr:`shellinertia` is false, otherwise its planar.
		"""
		return self._density


	@density.setter
	@blue.restrict
	def density(self, density: int|float|None) -> None:
		"""
		Unit defaults are set in SI units, but the physical properties can in principle be 
		interpreted in any system of measurements.

		Material density used to compute the geom mass and inertia. The computation is based on 
		the geom shape and the assumption of uniform density. The internal default of 1000 is the 
		density of water in SI units. This attribute is used only when the mass attribute above is 
		unspecified.
		
		Parameters
		----------
		density : int | float | None
			Density is either volumetric if :attr:`shellinertia` is false, otherwise its planar.
		"""
		self._density = float(density)


	@property
	def shellinertia(self) -> bool:
		"""
		If shellinertia is True all mass will be distributed along the surface of the object. 
		Changing this attribute also changes the effect of :attr:`density` from volumetric to planar.

		Returns
		-------
		bool
			Indicates whether mass is concentrated on the surface.
		"""
		return self._shellinertia


	@shellinertia.setter
	@blue.restrict
	def shellinertia(self, shellinertia: bool|None) -> None:
		"""
		If shellinertia is True all mass will be distributed along the surface of the object. 
		Changing this attribute also changes the effect of :attr:`density` from volumetric to planar.
		
		Parameters
		----------
		shellinertia : bool | None
			Indicates whether mass is concentrated on the surface.
		"""
		self._shellinertia = shellinertia


	@property
	def margin(self) -> float:
		"""
		Margin defines the minimum distance used to detect a contact.

		Returns
		-------
		float
			Default units are interpreted in meters.
		"""
		return self._margin


	@margin.setter
	@blue.restrict
	def margin(self, margin: int|float) -> None:
		"""
		Margin defines the minimum distance used to detect a contact.

		Parameters
		----------
		margin : int | float
			Default units are interpreted in meters.
		"""
		self._margin = float(margin)

	@property
	def gap(self) -> float:
		"""
		Gap defines the gap on minimum distance used to detect a contact.

		Returns
		-------
		float
			Default units are interpreted in meters.
		"""
		return self._gap


	@gap.setter
	@blue.restrict
	def gap(self, gap: int|float) -> None:
		"""
		Gap defines the gap on minimum distance used to detect a contact.

		Parameters
		----------
		gap : int | float
			Default units are interpreted in meters.
		"""
		self._gap = float(gap)


	@property
	def sliding_friction(self) -> float:
		"""
		Friction parameter for sliding used to computed the forces on contact pairs. 

		Returns
		-------
		float
			First component of :attr:`friction`.
		"""
		return self._sliding_friction


	@sliding_friction.setter
	@blue.restrict
	def sliding_friction(self, sliding_friction: int|float) -> None:
		"""
		Friction parameter for sliding used to computed the forces on contact pairs. 

		Parameters
		----------
		sliding_friction : int | float
			First component of :attr:`friction`.
		"""
		self._sliding_friction = float(sliding_friction)


	@property
	def torsional_friction(self) -> int|float:
		"""
		Friction parameter for torsion used to computed the forces on contact pairs. 

		Returns
		-------
		float
			Second component of :attr:`friction`.
		"""
		return self._torsional_friction


	@torsional_friction.setter
	@blue.restrict
	def torsional_friction(self, torsional_friction: int|float) -> None:
		"""
		Friction parameter for torsion used to computed the forces on contact pairs. 

		Parameters
		----------
		torsional_friction : int | float
			Second component of :attr:`friction`.
		"""
		self._torsional_friction = float(torsional_friction)


	@property
	def rolling_friction(self) -> int|float:
		"""
		Friction parameter for rolling used to computed the forces on contact pairs. 

		Returns
		-------
		float
			Third component of :attr:`friction`.
		"""
		return self._rolling_friction


	@rolling_friction.setter
	@blue.restrict
	def rolling_friction(self, rolling_friction: int|float) -> None:
		"""
		Friction parameter for rolling used to computed the forces on contact pairs. 

		Parameters
		----------
		rolling_friction : int | float
			Third component of :attr:`friction`.
		"""
		self._rolling_friction = float(rolling_friction)



class Capsule(blue.CapsuleGeomType, blue.tube.BaseTube, BaseGeom):

	"""
	Capsules consist of a cylinder with two half spheres at its ends. It can either be constructed from 
	its standard constructor for position, orientation, radius and length, or with the :meth:`blueprints.tube.BaseTube` 
	constructor. For a detailed description see :class:`blueprints.tube.BaseTube`.

	.. code-block:: python
		:caption: Standard constructor:

		>>> geom = blue.geoms.Capsule(alpha=TAU/8, length=3)
		>>> geom.head
		array([ 0.          1.06066017 -1.06066017], dtype=np.float32)
		>>> geom.tail
		array([ 0.         -1.06066017  1.06066017], dtype=np.float32)

	.. code-block:: python
		:caption: From points constructor:

		>>> geom = blue.geoms.Capsule.from_points([0, -1, 4], [2, 1, 0], radius=0.5)
		>>> geom.pos
		array([1. 0. 2.], dtype=np.float32)
		>>> geom.alpha, geom.beta, geom.gamma
		-2.6779450445889874 0.420534335283965 0.6847192030022825
		>>> geom.length
		4.898979485566356
	
	Attributes
	----------
	head : np.ndarra
		The first end of this Geoms. If it was defined by the :meth:`blueprints.tube.BaseTube` constructor, 
		``head`` is the first argument given.
	tail : np.ndarra
		The second end of this Geoms. If it was defined by the :meth:`blueprints.tube.BaseTube` constructor, 
		``tail`` is the second argument given.
	"""
	
	@blue.restrict
	def __init__(self, 
		     pos:      np.ndarray|list[int|float] = [0., 0., 0.],
		     radius:   int|float   = 1.,
		     length:   int|float   = 1.,
		     alpha:    int|float   = 0.,
		     beta:     int|float   = 0.,
		     gamma:    int|float   = 0., 
		     margin:   int|float   = 0.0,
		     gap:      int|float   = 0.0,
		     material: blue.MaterialType|None = None, 
		     color:    object|None = None, 
		     name:     str|None    = None, 
		     x:        int|float|np.int32|np.int64|np.float32|np.float64|None = None, 
		     y:        int|float|np.int32|np.int64|np.float32|np.float64|None = None, 
		     z:        int|float|np.int32|np.int64|np.float32|np.float64|None = None, 
		     quat:     list[int|float]|np.ndarray|None = None, 
		     **kwargs):
		"""
		Parameters
		----------
		pos : list[int | float] | np.ndarray | None, optional
			Represents the position of the object. Changing this attribute also changes the properties :attr:`x`, :attr:`y` and :attr:`z`.
		radius : int | float, optional
			Represents the radius of the cylinder and the two half spheres.
		length : int | float, optional
			The length of the cylinder.
		alpha : int | float | None, optional
			(Improper) euler angle of rotation around the x-axis in radian. Changing this value also changes the :attr:`euler` property.
		beta : int | float | None, optional
			(Improper) euler angle of rotation around the y-axis in radian. Changing this value also changes the :attr:`euler` property.
		gamma : int | float | None, optional
			(Improper) euler angle of rotation around the z-axis in radian. Changing this value also changes the :attr:`euler` property.
		margin : int | float | None, optional
			A contact is considered active only if the distance between the two geom surfaces is below 
			margin-gap.
		gap : int | float | None, optional
			This attribute is used to enable the generation of inactive contacts, i.e., contacts that are 
			ignored by the constraint solver but are included in mjData.contact for the purpose of custom 
			computations. When this value is positive, geom distances between margin and margin-gap 
			correspond to such inactive contacts.
		material : blue.MaterialType, optional
			The :class:`Material <blueprints.material.Material>` of the Geom. 
			:class:`Textures <blueprints.texture.BaseTexture> are applied via Materials.
		color : object
			The color of the Geom. See :class:`Color <blueprints.thing.colored.Color>` for a detailed description.
		name : str | None, optional
			The user specified name might potentially be altered to avoid a naming conflict by appending an enumeration scheme.
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
		self.radius = float(radius)
		self.length = float(length)
		super().__init__(pos=pos, 
				 alpha=alpha, 
				 beta=beta, 
				 gamma=gamma, 
				 material=material, 
				 margin=margin, 
				 gap=gap, 
				 color=color, 
				 name=name, 
				 x=x, 
				 y=y, 
				 z=z, 
				 quat=quat, 
				 **kwargs)


	@property
	def size(self) -> np.ndarray:
		"""
		.. note::
			In mujoco half lengths are used instead of true lengths of objects. This makes distance 
			calculations easier, but we find that it is overall more confusing then beneficial which
			is why blueprints uses proper lengths. Conversion is handled in the background, such that 
			users only need to specify true lengths.
		
		Returns
		-------
		np.ndarray
			The size defines the radius and half length of the capsule.
		"""
		return np.array([self.radius, 
				 self.length/2], dtype=np.float32)


	@size.setter
	@blue.restrict
	def size(self, size: np.ndarray|list[int|float]):
		"""
		.. note::
			In mujoco half lengths are used instead of true lengths of objects. This makes distance 
			calculations easier, but we find that it is overall more confusing then beneficial which
			is why blueprints uses proper lengths. Conversion is handled in the background, such that 
			users only need to specify true lengths.
		
		Parameters
		----------
		size : np.ndarray | list[int | float]
			The size defines the radius and half length of the capsule.
		"""
		self.radius = float(size[0])
		self.length = float(size[1]) * 2



class Cylinder(blue.CylinderGeomType, blue.tube.BaseTube, BaseGeom):

	"""
	Cylinders can either be constructed from their standard constructor for position, orientation, 
	radius and length, or with the :meth:`blueprints.tube.BaseTube` constructor. For a detailed description 
	see :class:`blueprints.tube.BaseTube`.

	.. code-block:: python
		:caption: Standard constructor:

		>>> geom = blue.geoms.Cylinder(alpha=TAU/8, length=3)
		>>> geom.head
		array([ 0.          1.06066017 -1.06066017], dtype=np.float32)
		>>> geom.tail
		array([ 0.         -1.06066017  1.06066017], dtype=np.float32)

	.. code-block:: python
		:caption: From points constructor:

		>>> geom = blue.geoms.Cylinder.from_points([0, -1, 4], [2, 1, 0], radius=0.5)
		>>> geom.pos
		array([1. 0. 2.], dtype=np.float32)
		>>> geom.alpha, geom.beta, geom.gamma
		-2.6779450445889874 0.420534335283965 0.6847192030022825
		>>> geom.length
		4.898979485566356
	
	Attributes
	----------
	head : np.ndarra
		The first end of this Geoms. If it was defined by the :meth:`blueprints.tube.BaseTube` constructor, 
		``head`` is the first argument given.
	tail : np.ndarra
		The second end of this Geoms. If it was defined by the :meth:`blueprints.tube.BaseTube` constructor, 
		``tail`` is the second argument given.
	"""
	
	@blue.restrict
	def __init__(self, 
		     pos:      np.ndarray|list[int|float] = [0., 0., 0.],
		     radius:   int|float   = 1.,
		     length:   int|float   = 1.,
		     alpha:    int|float   = 0.,
		     beta:     int|float   = 0.,
		     gamma:    int|float   = 0., 
		     margin:   int|float   = 0.0,
		     gap:      int|float   = 0.0,
		     material: blue.MaterialType|None = None, 
		     color:    object|None = None, 
		     name:     str|None    = None, 
		     x:        int|float|np.int32|np.int64|np.float32|np.float64|None = None, 
		     y:        int|float|np.int32|np.int64|np.float32|np.float64|None = None, 
		     z:        int|float|np.int32|np.int64|np.float32|np.float64|None = None, 
		     quat:     list[int|float]|np.ndarray|None = None, 
		     **kwargs):
		"""
		Parameters
		----------
		pos : list[int | float] | np.ndarray | None, optional
			Represents the position of the object. Changing this attribute also changes the properties :attr:`x`, :attr:`y` and :attr:`z`.
		radius : int | float, optional
			Represents the radius of the cylinder and the two half spheres.
		length : int | float, optional
			The length of the cylinder.
		alpha : int | float | None, optional
			(Improper) euler angle of rotation around the x-axis in radian. Changing this value also changes the :attr:`euler` property.
		beta : int | float | None, optional
			(Improper) euler angle of rotation around the y-axis in radian. Changing this value also changes the :attr:`euler` property.
		gamma : int | float | None, optional
			(Improper) euler angle of rotation around the z-axis in radian. Changing this value also changes the :attr:`euler` property.
		margin : int | float | None, optional
			A contact is considered active only if the distance between the two geom surfaces is below 
			margin-gap.
		gap : int | float | None, optional
			This attribute is used to enable the generation of inactive contacts, i.e., contacts that are 
			ignored by the constraint solver but are included in mjData.contact for the purpose of custom 
			computations. When this value is positive, geom distances between margin and margin-gap 
			correspond to such inactive contacts.
		material : blue.MaterialType, optional
			The :class:`Material <blueprints.material.Material>` of the Geom. 
			:class:`Textures <blueprints.texture.BaseTexture> are applied via Materials.
		color : object
			The color of the Geom. See :class:`Color <blueprints.thing.colored.Color>` for a detailed description.
		name : str | None, optional
			The user specified name might potentially be altered to avoid a naming conflict by appending an enumeration scheme.
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
		self.radius = float(radius)
		self.length = float(length)
		super().__init__(pos=pos, 
				 alpha=alpha, 
				 beta=beta, 
				 gamma=gamma, 
				 margin=margin, 
				 gap=gap, 
				 material=material, 
				 color=color, 
				 name=name, 
				 x=x, 
				 y=y, 
				 z=z, 
				 quat=quat, 
				 **kwargs)


	@property
	def size(self) -> np.ndarray:
		"""
		.. note::
			In mujoco half lengths are used instead of true lengths of objects. This makes distance 
			calculations easier, but we find that it is overall more confusing then beneficial which
			is why blueprints uses proper lengths. Conversion is handled in the background, such that 
			users only need to specify true lengths.
		
		Returns
		-------
		np.ndarray
			The size defines the radius and half length of the cylinder.
		"""
		return np.array([self.radius, 
				 self.length/2], dtype=np.float32)


	@size.setter
	@blue.restrict
	def size(self, size: np.ndarray|list[int|float]):
		"""
		.. note::
			In mujoco half lengths are used instead of true lengths of objects. This makes distance 
			calculations easier, but we find that it is overall more confusing then beneficial which
			is why blueprints uses proper lengths. Conversion is handled in the background, such that 
			users only need to specify true lengths.
		
		Parameters
		----------
		size : np.ndarray | list[int | float]
			 The size defines the radius and half length of the cylinder.
		"""
		self.radius = float(size[0])
		self.length = float(size[1]) * 2



class Box(blue.BoxGeomType, blue.tube.BaseTube, BaseGeom):

	"""
	Boxes can either be constructed from their standard constructor for position, orientation, 
	and x-, y-, z-length, or with the :meth:`blueprints.tube.BaseTube` constructor. For a detailed 
	description see :class:`blueprints.tube.BaseTube`.

	.. code-block:: python
		:caption: Standard constructor:

		>>> geom = blue.geoms.Box(x_length=2, y_length=4, z_length=1)
		>>> geom.head
		array([ 0.   0.  -0.5], dtype=np.float32)
		>>> geom.tail
		array([ 0.   0.   0.5], dtype=np.float32)

	.. code-block:: python
		:caption: From points constructor:

		>>> geom = blue.geoms.Box.from_points([4, 0,-2], [2, 4, 0], radius=0.5)
		>>> geom.pos
		array([ 3.   2.  -1.], dtype=np.float32)
		>>> geom.alpha, geom.beta, geom.gamma
		-1.1071487177940904 -0.4205343352839653 0.20135792079032988
		>>> geom.x_length, geom.y_length, geom.z_length
		0.5 0.5 4.898979485566356
	
	Attributes
	----------
	head : np.ndarra
		The first end of this Geoms. If it was defined by the :meth:`blueprints.tube.BaseTube` constructor, 
		``head`` is the first argument given.
	tail : np.ndarra
		The second end of this Geoms. If it was defined by the :meth:`blueprints.tube.BaseTube` constructor, 
		``tail`` is the second argument given.
	"""
	
	@blue.restrict
	def __init__(self, 
		     pos:      np.ndarray|list[int|float] = [0., 0., 0.], 
		     x_length: int|float   = 1., 
		     y_length: int|float   = 1., 
		     z_length: int|float   = 1., 
		     alpha:    int|float   = 0., 
		     beta:     int|float   = 0., 
		     gamma:    int|float   = 0.,  
		     margin:   int|float   = 0.0,
		     gap:      int|float   = 0.0,
		     material: blue.MaterialType|None = None, 
		     color:    object|None = None, 
		     name:     str|None    = None, 
		     x:        int|float|np.int32|np.int64|np.float32|np.float64|None = None, 
		     y:        int|float|np.int32|np.int64|np.float32|np.float64|None = None, 
		     z:        int|float|np.int32|np.int64|np.float32|np.float64|None = None, 
		     quat:     list[int|float]|np.ndarray|None = None, 
		     **kwargs):
		"""
		Parameters
		----------
		pos : list[int | float] | np.ndarray | None, optional
			Represents the position of the object. Changing this attribute also changes the properties :attr:`x`, :attr:`y` and :attr:`z`.
		x_length : int | float, optional
			The true length of the Box along the x-axis.
		y_length : int | float, optional
			The true length of the Box along the y-axis.
		z_length : int | float, optional
			The true length of the Box along the z-axis.
		alpha : int | float | None, optional
			(Improper) euler angle of rotation around the x-axis in radian. Changing this value also changes the :attr:`euler` property.
		beta : int | float | None, optional
			(Improper) euler angle of rotation around the y-axis in radian. Changing this value also changes the :attr:`euler` property.
		gamma : int | float | None, optional
			(Improper) euler angle of rotation around the z-axis in radian. Changing this value also changes the :attr:`euler` property.
		margin : int | float | None, optional
			A contact is considered active only if the distance between the two geom surfaces is below 
			margin-gap.
		gap : int | float | None, optional
			This attribute is used to enable the generation of inactive contacts, i.e., contacts that are 
			ignored by the constraint solver but are included in mjData.contact for the purpose of custom 
			computations. When this value is positive, geom distances between margin and margin-gap 
			correspond to such inactive contacts.
		material : blue.MaterialType, optional
			The :class:`Material <blueprints.material.Material>` of the Geom. 
			:class:`Textures <blueprints.texture.BaseTexture> are applied via Materials.
		color : object
			The color of the Geom. See :class:`Color <blueprints.thing.colored.Color>` for a detailed description.
		name : str | None, optional
			The user specified name might potentially be altered to avoid a naming conflict by appending an enumeration scheme.
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
		self.x_length = float(x_length)
		self.y_length = float(y_length)
		self.z_length = float(z_length)
		super().__init__(pos=pos, 
				 alpha=alpha, 
				 beta=beta, 
				 gamma=gamma, 
				 margin=margin, 
				 gap=gap, 
				 material=material, 
				 color=color, 
				 name=name, 
				 x=x, 
				 y=y, 
				 z=z, 
				 quat=quat, 
				 **kwargs)


	@property
	def size(self) -> np.ndarray:
		"""
		.. note::
			In mujoco half lengths are used instead of true lengths of objects. This makes distance 
			calculations easier, but we find that it is overall more confusing then beneficial which
			is why blueprints uses proper lengths. Conversion is handled in the background, such that 
			users only need to specify true lengths.
		
		Returns
		-------
		np.ndarray
			The size defines the half lengths of the Box.
		"""
		return np.array([self.x_length/2, 
				 self.y_length/2, 
				 self.z_length/2], dtype=np.float32)


	@size.setter
	@blue.restrict
	def size(self, size: np.ndarray|list[int|float]):
		"""
		.. note::
			In mujoco half lengths are used instead of true lengths of objects. This makes distance 
			calculations easier, but we find that it is overall more confusing then beneficial which
			is why blueprints uses proper lengths. Conversion is handled in the background, such that 
			users only need to specify true lengths.
		
		Parameters
		----------
		size : np.ndarray | list[int | float]
			The size defines the half lengths of the Box.
		"""
		self.x_length = float(size[0]) * 2
		self.y_length = float(size[1]) * 2
		self.z_length = float(size[2]) * 2



class Plane(blue.PlaneGeomType, BaseGeom):

	"""
	Planes can either be finite or infinite. They are normal to the Z-axis of their frame of reference 
	and the outside of the plane lies in the positive Z-axis. The inside of the plane is rendered semi 
	transparent. Finite Planes are rendered as rectangles with size specified in :attr:`x_length` 
	:attr:`y_length`. If those size parameters are set to zero or are set negative, the plain is instead 
	infinite. The :attr:`spacing` defines the size of grid subdivisions which are used in rendering.
	"""
	
	@blue.restrict
	def __init__(self,
		     pos:      np.ndarray|list[int|float] = [0., 0., 0.], 
		     x_length: int|float   = 0., 
		     y_length: int|float   = 0., 
		     spacing:  int|float   = 1., 
		     alpha:    int|float   = 0., 
		     beta:     int|float   = 0., 
		     gamma:    int|float   = 0., 
		     margin:   int|float   = 0.0,
		     gap:      int|float   = 0.0,
		     material: blue.MaterialType|None = None, 
		     color:    object|None = None, 
		     name:     str|None    = None, 
		     x:        int|float|np.int32|np.int64|np.float32|np.float64|None = None, 
		     y:        int|float|np.int32|np.int64|np.float32|np.float64|None = None, 
		     z:        int|float|np.int32|np.int64|np.float32|np.float64|None = None, 
		     quat:     list[int|float]|np.ndarray|None = None, 
		     **kwargs):
		"""
		Parameters
		----------
		pos : list[int | float] | np.ndarray | None, optional
			Represents the position of the object. Changing this attribute also changes the properties :attr:`x`, :attr:`y` and :attr:`z`.
		x_length : int | float, optional
			Length of the X-axis side. If both lengths are set to zero or negative the Plane is infinite.
		y_length : int | float, optional
			Length of the Y-axis side. If both lengths are set to zero or negative the Plane is infinite.
		spacing : int | float, optional
			Spacing between the grid subdivisions.
		alpha : int | float | None, optional
			(Improper) euler angle of rotation around the x-axis in radian. Changing this value also changes the :attr:`euler` property.
		beta : int | float | None, optional
			(Improper) euler angle of rotation around the y-axis in radian. Changing this value also changes the :attr:`euler` property.
		gamma : int | float | None, optional
			(Improper) euler angle of rotation around the z-axis in radian. Changing this value also changes the :attr:`euler` property.
		margin : int | float | None, optional
			A contact is considered active only if the distance between the two geom surfaces is below 
			margin-gap.
		gap : int | float | None, optional
			This attribute is used to enable the generation of inactive contacts, i.e., contacts that are 
			ignored by the constraint solver but are included in mjData.contact for the purpose of custom 
			computations. When this value is positive, geom distances between margin and margin-gap 
			correspond to such inactive contacts.
		material : blue.MaterialType, optional
			The :class:`Material <blueprints.material.Material>` of the Geom. 
			:class:`Textures <blueprints.texture.BaseTexture> are applied via Materials.
		color : object
			The color of the Geom. See :class:`Color <blueprints.thing.colored.Color>` for a detailed description.
		name : str | None, optional
			The user specified name might potentially be altered to avoid a naming conflict by appending an enumeration scheme.
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
		self.x_length = float(x_length)
		self.y_length = float(y_length)
		self.spacing  = float(spacing)
		super().__init__(pos=pos, 
				 alpha=alpha, 
				 beta=beta, 
				 gamma=gamma, 
				 margin=margin, 
				 gap=gap, 
				 material=material, 
				 color=color, 
				 name=name, 
				 x=x, 
				 y=y, 
				 z=z, 
				 quat=quat, 
				 **kwargs)


	@property
	def size(self) -> np.ndarray:
		"""
		.. note::
			In mujoco half lengths are used instead of true lengths of objects. This makes distance 
			calculations easier, but we find that it is overall more confusing then beneficial which
			is why blueprints uses proper lengths. Conversion is handled in the background, such that 
			users only need to specify true lengths.
		
		Returns
		-------
		np.ndarray
			The first two components are half lengths for the X-axis and the Y-axis and the third is the spacing between grid subdivisions.
		"""
		return np.array([self.x_length/2, 
				 self.y_length/2, 
				 self.spacing], dtype=np.float32)


	@size.setter
	@blue.restrict
	def size(self, size: np.ndarray|list[int|float]):
		"""
		.. note::
			In mujoco half lengths are used instead of true lengths of objects. This makes distance 
			calculations easier, but we find that it is overall more confusing then beneficial which
			is why blueprints uses proper lengths. Conversion is handled in the background, such that 
			users only need to specify true lengths.
		
		Parameters
		----------
		size : np.ndarray | list[int | float]
			The first two components are half lengths for the X-axis and the Y-axis and the third is the spacing between grid subdivisions.
		"""
		self.x_length = float(size[0]) * 2
		self.y_length = float(size[1]) * 2
		self.spacing  = float(size[2])



class Sphere(blue.SphereGeomType, BaseGeom):

	"""
	Spheres are defined via radius and position.
	
	Attributes
	----------
	radius : float
		The radius of the sphere
	"""
	
	@blue.restrict
	def __init__(self, 
		     pos:    np.ndarray|list[int|float] = [0., 0., 0.],
		     radius: int|float   = 1.,
		     alpha:  int|float   = 0.,
		     beta:   int|float   = 0.,
		     gamma:  int|float   = 0., 
		     margin: int|float   = 0.0,
		     gap:    int|float   = 0.0,
		     material: blue.MaterialType|None = None, 
		     color:  object|None = None, 
		     name:   str|None    = None, 
		     x:      int|float|np.int32|np.int64|np.float32|np.float64|None = None, 
		     y:      int|float|np.int32|np.int64|np.float32|np.float64|None = None, 
		     z:      int|float|np.int32|np.int64|np.float32|np.float64|None = None, 
		     quat:   list[int|float]|np.ndarray|None = None, 
		     **kwargs):
		"""
		Parameters
		----------
		pos : list[int | float] | np.ndarray | None, optional
			Represents the position of the Thing. Changing this attribute also changes the properties :attr:`x`, :attr:`y` and :attr:`z`.
		radius : int | float, optional
			The radius of the Sphere.
		alpha : int | float | None, optional
			(Improper) euler angle of rotation around the x-axis in radian. Changing this value also changes the :attr:`euler` property.
		beta : int | float | None, optional
			(Improper) euler angle of rotation around the y-axis in radian. Changing this value also changes the :attr:`euler` property.
		gamma : int | float | None, optional
			(Improper) euler angle of rotation around the z-axis in radian. Changing this value also changes the :attr:`euler` property.
		margin : int | float | None, optional
			A contact is considered active only if the distance between the two geom surfaces is below 
			margin-gap.
		gap : int | float | None, optional
			This attribute is used to enable the generation of inactive contacts, i.e., contacts that are 
			ignored by the constraint solver but are included in mjData.contact for the purpose of custom 
			computations. When this value is positive, geom distances between margin and margin-gap 
			correspond to such inactive contacts.
		material : blue.MaterialType, optional
			The :class:`Material <blueprints.material.Material>` of the Geom. 
			:class:`Textures <blueprints.texture.BaseTexture> are applied via Materials.
		color : object
			The color of the Geom. See :class:`Color <blueprints.thing.colored.Color>` for a detailed description.
		name : str | None, optional
			The user specified name might potentially be altered to avoid a naming conflict by appending an enumeration scheme.
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
		self.radius = float(radius)
		super().__init__(pos=pos, 
				 alpha=alpha, 
				 beta=beta, 
				 gamma=gamma, 
				 margin=margin, 
				 gap=gap, 
				 material=material, 
				 color=color, 
				 name=name, 
				 x=x, 
				 y=y, 
				 z=z, 
				 quat=quat, 
				 **kwargs)


	@property
	def size(self) -> np.ndarray:
		"""
		.. note::
			In mujoco half lengths are used instead of true lengths of objects. This makes distance 
			calculations easier, but we find that it is overall more confusing then beneficial which
			is why blueprints uses proper lengths. Conversion is handled in the background, such that 
			users only need to specify true lengths.
		
		Returns
		-------
		np.ndarray
			The only component of size is the radius, which is interpreted as meters by default parameters and convention.
		"""
		return np.array([self.radius])


	@size.setter
	@blue.restrict
	def size(self, size: np.ndarray|list[int|float]):
		"""
		.. note::
			In mujoco half lengths are used instead of true lengths of objects. This makes distance 
			calculations easier, but we find that it is overall more confusing then beneficial which
			is why blueprints uses proper lengths. Conversion is handled in the background, such that 
			users only need to specify true lengths.
		
		Parameters
		----------
		size : np.ndarray | list[int | float]
			The only component of size is the radius, which is interpreted as meters by default parameters and convention.
		"""
		self.radius = float(size[0])



class Ellipsoid(blue.EllipsoidGeomType, BaseGeom):

	"""
	Ellipsoids are defined with via :attr:´x_radius`, :attr:´y_radius` and :attr:´z_radius`. The surface of 
	the Ellipsoid contains the solutions to the equation:

	:math:`\\left( \\frac{x}{r_x} \\right)^2 + \\left( \\frac{y}{r_y} \\right)^2 + \\left( \\frac{z}{r_z} \\right)^2 = 1`
	
	Attributes
	----------
	x_radius : float
		This attribute defines the radius for the X-axis.
	y_radius : float
		This attribute defines the radius for the Y-axis.
	z_radius : float
		This attribute defines the radius for the Z-axis.
	"""
	
	@blue.restrict
	def __init__(self, 
		     pos:      np.ndarray|list[int|float] = [0., 0., 0.],
		     x_radius: int|float   = 1.,
		     y_radius: int|float   = 2.,
		     z_radius: int|float   = 3.,
		     alpha:    int|float   = 0.,
		     beta:     int|float   = 0.,
		     gamma:    int|float   = 0., 
		     margin:   int|float   = 0.0,
		     gap:      int|float   = 0.0,
		     material: blue.MaterialType|None = None, 
		     color:    object|None = None, 
		     name:     str|None    = None, 
		     x:        int|float|np.int32|np.int64|np.float32|np.float64|None = None, 
		     y:        int|float|np.int32|np.int64|np.float32|np.float64|None = None, 
		     z:        int|float|np.int32|np.int64|np.float32|np.float64|None = None, 
		     quat:     list[int|float]|np.ndarray|None = None, 
		     **kwargs):
		"""
		Parameters
		----------
		pos : list[int | float] | np.ndarray | None, optional
			Represents the position of the object. Changing this attribute also changes the properties :attr:`x`, :attr:`y` and :attr:`z`.
		x_radius : int | float, optional
			This attribute defines the radius for the X-axis.
		y_radius : int | float, optional
			This attribute defines the radius for the Y-axis.
		z_radius : int | float, optional
			This attribute defines the radius for the Z-axis.
		alpha : int | float | None, optional
			(Improper) euler angle of rotation around the x-axis in radian. Changing this value also changes the :attr:`euler` property.
		beta : int | float | None, optional
			(Improper) euler angle of rotation around the y-axis in radian. Changing this value also changes the :attr:`euler` property.
		gamma : int | float | None, optional
			(Improper) euler angle of rotation around the z-axis in radian. Changing this value also changes the :attr:`euler` property.
		margin : int | float | None, optional
			A contact is considered active only if the distance between the two geom surfaces is below 
			margin-gap.
		gap : int | float | None, optional
			This attribute is used to enable the generation of inactive contacts, i.e., contacts that are 
			ignored by the constraint solver but are included in mjData.contact for the purpose of custom 
			computations. When this value is positive, geom distances between margin and margin-gap 
			correspond to such inactive contacts.
		material : blue.MaterialType, optional
			The :class:`Material <blueprints.material.Material>` of the Geom. 
			:class:`Textures <blueprints.texture.BaseTexture> are applied via Materials.
		color : object
			The color of the Geom. See :class:`Color <blueprints.thing.colored.Color>` for a detailed description.
		name : str | None, optional
			The user specified name might potentially be altered to avoid a naming conflict by appending an enumeration scheme.
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
		self.x_radius = float(x_radius)
		self.y_radius = float(y_radius)
		self.z_radius = float(z_radius)
		super().__init__(pos=pos, 
				 alpha=alpha, 
				 beta=beta, 
				 gamma=gamma, 
				 margin=margin, 
				 gap=gap, 
				 material=material, 
				 color=color, 
				 name=name, 
				 x=x, 
				 y=y, 
				 z=z, 
				 quat=quat, 
				 **kwargs)


	@property
	def size(self) -> np.ndarray:
		"""
		.. note::
			In mujoco half lengths are used instead of true lengths of objects. This makes distance 
			calculations easier, but we find that it is overall more confusing then beneficial which
			is why blueprints uses proper lengths. Conversion is handled in the background, such that 
			users only need to specify true lengths.
		
		Returns
		-------
		np.ndarray
			The components contain the :attr:`x_radius`, :attr:`y_radius` and :attr:`z_radius` attribute.
		"""
		return np.array([self.x_radius, 
				 self.y_radius, 
				 self.z_radius], dtype=np.float32)


	@size.setter
	@blue.restrict
	def size(self, size: np.ndarray|list[int|float]):
		"""
		.. note::
			In mujoco half lengths are used instead of true lengths of objects. This makes distance 
			calculations easier, but we find that it is overall more confusing then beneficial which
			is why blueprints uses proper lengths. Conversion is handled in the background, such that 
			users only need to specify true lengths.
		
		Parameters
		----------
		size : np.ndarray | list[int | float]
			The components contain the :attr:`x_radius`, :attr:`y_radius` and :attr:`z_radius` attribute.
		"""
		self.x_radius = float(size[0])
		self.y_radius = float(size[1])
		self.z_radius = float(size[2])



class Mesh(blue.MeshGeomType, BaseGeom):

	"""
	Meshes can be used to build more complex shapes either by specifying vertecies manually or by loading 
	mesh files directly.

	>>> blue.geoms.Mesh(filename='utahteapot.stl')
	Mesh<unnamed_mesh:1>
	>>> vertecies = [[0, 0, 0], 
		         [0, 2, 0], 
		         [2, 2, 0], 
		         [2, 0, 0], 
		         [1, 1, 2]]
	>>> mesh = blue.geoms.Mesh(vertecies=vertecies, centered=True)
	>>> mesh.vertecies
	[[-1. -1. -1.]
	 [-1.  1. -1.]
	 [ 1.  1. -1.]
	 [ 1. -1. -1.]
	 [ 0.  0.  1.]]


	The vertex data is accessed through a :class:`blueprints.assets.MeshAsset` which 
	in turn stores the data in a :class:`blueprints.cache.MeshCache`. As long as no vertex data are 
	modified copies of the :class:`Mesh` will reference the same Asset to avoid redundancy, otherwise a 
	new Asset and or Cache is created. The creation of a new Asset is computationally fairly cheap, the 
	creation of a new Cache may come with significant costs if the mesh file is large.

	Modifications of the following attributes might trigger the creation of a new Asset:

	1. ``pos``
	2. ``euler``
	3. ``scale``

	Modifications of the following attributes might trigger the creation of a new Cache:

	1. ``size``
	2. ``vertecies``


	Raises
	------
	Exception
		If neither a filename, vertecies or an asset are given an error is raised.
	"""

	@blue.restrict
	def __init__(self, 
		     pos:       np.ndarray|list[int|float]            = [0., 0., 0.],
		     vertecies: list[np.ndarray|list[float|int]]|None = None,  
		     filename:  str|None                              = None, 
		     centered:  bool                                  = True, 
		     asset:     blue.assets.MeshAsset|None            = None, 
		     material:  blue.MaterialType|None                = None, 
		     margin:    int|float   = 0.0,
		     gap:       int|float   = 0.0,
		     x:         int|float|np.int32|np.int64|np.float32|np.float64|None = None, 
		     y:         int|float|np.int32|np.int64|np.float32|np.float64|None = None, 
		     z:         int|float|np.int32|np.int64|np.float32|np.float64|None = None, 
		     **kwargs) -> None:
		"""
		Parameters
		----------
		pos : list[int | float] | np.ndarray | None, optional
			Represents the position of the object. Changing this attribute also changes the properties :attr:`x`, :attr:`y` and :attr:`z`.
		vertecies : list[np.ndarray | list[float | int]] | None, optional
			Either a list of positions of vertecies or None.
		filename : str | None, optional
			The name for the file from which the Mesh data is loaded. The supported formats are ``.stl`` in binary and ``.obj`` in binary and ascii.
		centered : bool, optional
			If this argument is set, vertecies are centered around the references frames origin.
		asset : blue.assets.MeshAsset | None, optional
			A possible asset, from which the Mesh is constructed.
		margin : int | float | None, optional
			A contact is considered active only if the distance between the two geom surfaces is below 
			margin-gap.
		gap : int | float | None, optional
			This attribute is used to enable the generation of inactive contacts, i.e., contacts that are 
			ignored by the constraint solver but are included in mjData.contact for the purpose of custom 
			computations. When this value is positive, geom distances between margin and margin-gap 
			correspond to such inactive contacts.
		material : blue.MaterialType, optional
			The :class:`Material <blueprints.material.Material>` of the Geom. 
			:class:`Textures <blueprints.texture.BaseTexture> are applied via Materials.
		x : int | float |np.int32 | np.int64 | np.float32 | np.float64 | None, optional
			If `pos` is not specified, this argument sets the X position coordinate.
		y : int | float |np.int32 | np.int64 | np.float32 | np.float64 | None, optional
			If `pos` is not specified, this argument sets the Y position coordinate.
		z : int | float |np.int32 | np.int64 | np.float32 | np.float64 | None, optional
			If `pos` is not specified, this argument sets the Z position coordinate.
		**kwargs
			Keyword arguments are passed to ``super().__init__``.
		
		Raises
		------
		Exception
			If neither a filename, vertecies or an asset are given an error is raised.
		"""
		kwargs_nameless = kwargs.copy()
		if 'name' in kwargs_nameless:
			del kwargs_nameless['name']
		if sum(x is not None for x in (vertecies, filename, asset)) > 1:
			raise Exception('Not more than one argument (vertecies, filename or asset) is allowed to be None.')
		if pos is None:
			x = float(x) if x is not None else 0.
			y = float(y) if y is not None else 0.
			z = float(z) if z is not None else 0.
			pos = np.array([x, y, z], dtype=np.float32)
		if asset is not None:
			self.asset = asset
		elif filename is not None:
			if not os.path.isfile(filename):
				# TRY DIRNAME PREFIX TO RESOLVE RELATIVE REF TO MAIN.PY
				filename = f'{os.path.dirname(sys.argv[0])}/{filename}'
				if not os.path.isfile(filename):
					raise Exception(f'File not found for path {filename}')
			if not os.path.isabs(filename):
				filename = os.path.abspath(filename)
			if not os.path.isabs(filename):
				path = os.path.abspath(os.path.dirname(sys.argv[0]))
				filename = f'{path}/{filename}'
			self.asset = blue.assets.MeshAsset(filename=filename, 
							   pos=pos, 
							   centered=centered, 
							   xml_data=vertecies is not None, 
							   **kwargs_nameless)
		elif vertecies is not None:
			self.asset = blue.assets.MeshAsset(vertecies=vertecies, 
							   pos=pos, 
							   centered=centered, 
							   xml_data=True, 
							   **kwargs_nameless)
		else:
			raise Exception('No asset could be constructed! Please provide either a filename, vertcies or an asset.')
		# SEPARATE CHANGES FOR POST-FREEZE ASSIGNMENT
		self.asset.freeze = True
		kwchanges   = {}
		if len(self.asset._references) > 0:
			proto_parent = next(iter(self.asset._references))
			for key, val in kwargs.items():
				if hasattr(proto_parent, key):
					equal = getattr(proto_parent, key) == val
					equal = bool(np.all(equal)) if isinstance(equal, np.ndarray) else equal
					if not equal:
						kwchanges[key] = val
						kwargs[key] = getattr(proto_parent, key)
			if not bool(np.all(proto_parent.pos == pos)):
				kwchanges['pos'] = pos
				pos = proto_parent.pos
		self.asset._add(self)
		#print(self.copy)
		super().__init__(pos=pos, 
				 material=material, 
				 margin=margin, 
				 gap=gap, 
				 **kwargs)
		self.asset.freeze = False
		for key, val in kwchanges.items():
			if key in self._BLUEPRINT_ATTR():
				setattr(self, key, val)


	@blue.restrict
	def _build(self, 
		   parent, 
		   world, 
		   indicies, 
		   **kwargs):
		"""
		This method is called to build the xml.
		
		Parameters
		----------
		parent : xml.etree.ElementTree.Element
			The xml element of its parent
		world : WorldType
			The World from which the build method was called initially
		
		Returns
		-------
		xml.etree.ElementTree.Element
			The builded xml element of the Thing.
		"""
		if self.material is not None:
			self.material._build(parent, world, indicies, **kwargs)
			kwargs['material'] = self.material.asset.name
		self._xml_root = xml.SubElement(parent, 
						self._MUJOCO_OBJ, 
						mesh=self.asset.name, 
						**self._mujoco_specs(kwargs))
		if self.asset._built:
			self._index = self.asset._index
		else:
			self._index       = indicies['mesh']
			indicies['mesh'] += 1
		self.asset._build(parent=parent, 
				  world=world, 
				  indicies=indicies, 
				  **kwargs)
		return self._xml_root


	@blue.restrict
	@classmethod
	def _from_xml_element(cls, 
			      xml_element: xml.Element, 
			      asset:       blue.AssetType) -> blue.ThingType:
		"""
		This method reconstructs a Mesh from an xml element.
		
		Parameters
		----------
		xml_element : xml.Element
			The xml element from which a Mesh is reconstructed.
		asset : blue.AssetType
			The asset from which the Mesh takes its data.
		
		Returns
		-------
		blue.ThingType
			The reconstructed Mesh.
		"""
		init_args, post_args, rest_args = cls._xml_element_args(xml_element)
		geom_type = rest_args['type']
		geom = object.__new__(blue.REGISTER.GEOM_THINGS[geom_type])
		init_args['asset'] = asset
		geom.__init__(**init_args)
		for key, val in post_args.items():
			setattr(geom, key, val)
		return geom


	@property
	def vertecies(self) -> list[np.ndarray]:
		"""
		Modifying this attribute will create a new :class:`blueprints.cache.MeshCache` object if 
		the previous Cache was the children of multiple parents. Otherwise the Cache get modified 
		directly. 
		
		Returns
		-------
		list[np.ndarray]
			A list of all vertex positions.
		"""
		return self.asset.vertecies.copy()


	@vertecies.setter
	@blue.restrict
	def vertecies(self, vertecies: np.ndarray|list[np.ndarray|list[float|int]]) -> None:
		"""
		Modifying this attribute will create a new :class:`blueprints.cache.MeshCache` object if 
		the previous Cache was the children of multiple parents. Otherwise the Cache get modified 
		directly. 
		
		Parameters
		----------
		vertecies : np.ndarray | list[np.ndarray | list[float | int]]
			A list of all vertex positions.
		
		"""
		self.asset._prepare_for_modification(self)
		self.asset.vertecies = vertecies


	@property
	def pos(self) -> np.ndarray:
		"""
		Modifying this attribute will create a new :class:`blueprints.assets.MeshAsset` object if 
		the previous Asset was the children of multiple parents. Otherwise the Asset get modified 
		directly. 
		
		Returns
		-------
		np.ndarray | list[float | int]
		"""
		return self.asset.pos.copy()


	@pos.setter
	@blue.restrict
	def pos(self, pos: np.ndarray|list[float|int]) -> None:
		"""
		Modifying this attribute will create a new :class:`blueprints.assets.MeshAsset` object if 
		the previous Asset was the children of multiple parents. Otherwise the Asset get modified 
		directly. 
		
		Parameters
		----------
		pos : np.ndarray | list[float | int]
			Description
		"""
		self.asset._prepare_for_modification(self)
		self.asset.pos = pos


	@property
	def _pos(self) -> np.ndarray:
		"""
		Returns
		-------
		np.ndarray | list[float | int]
		"""
		return self.asset._pos.copy()


	@_pos.setter
	@blue.restrict
	def _pos(self, pos: np.ndarray|list[float|int]) -> None:
		"""
		Parameters
		----------
		pos : np.ndarray | list[float | int]
		"""
		self.asset._prepare_for_modification(self)
		self.asset._pos = pos


	@property
	def euler(self) -> np.ndarray:
		"""
		Modifying this attribute will create a new :class:`blueprints.assets.MeshAsset` object if 
		the previous Asset was the children of multiple parents. Otherwise the Asset get modified 
		directly. 
		
		Returns
		-------
		np.ndarray
		"""
		return self.asset.euler


	@euler.setter
	@blue.restrict
	def euler(self, euler: np.ndarray|list[float|int]) -> None:
		"""
		Modifying this attribute will create a new :class:`blueprints.assets.MeshAsset` object if 
		the previous Asset was the children of multiple parents. Otherwise the Asset get modified 
		directly. 
		
		Parameters
		----------
		euler : np.ndarray | list[float | int]
			Description
		
		Deleted Parameters
		------------------
		vertex : list[np.ndarray | list[float | int]]
		Description
		pos : np.ndarray | list[float | int]
		Description
		"""
		self.asset._prepare_for_modification(self)
		self.asset.euler = euler


	@property
	def size(self) -> np.ndarray:
		"""
		Modifying this attribute will create a new :class:`blueprints.cache.MeshCache` object if 
		the previous Cache was the children of multiple parents. Otherwise the Cache get modified 
		directly. 
		
		.. note::
			Altering this attribute also changes the :attr:`vertecies` attribute. To get the same 
			effect without changing the vertecies, see :attr:`scale`.
		
		Returns
		-------
		np.ndarray
			The size of the Mesh is calculated as the size of the Box with edges along the axes of 
			the Meshes frame of reference, that captures all vertecies.
		"""
		return self.asset.size


	@size.setter
	@blue.restrict
	def size(self, size: np.ndarray|list[float|int]) -> None:
		"""
		Modifying this attribute will create a new :class:`blueprints.cache.MeshCache` object if 
		the previous Cache was the children of multiple parents. Otherwise the Cache get modified 
		directly.

		.. note::
			Altering this attribute also changes the :attr:`vertecies` attribute. To get the same 
			effect without changing the vertecies, see :attr:`scale`.
		
		Parameters
		----------
		size : np.ndarray | list[float | int]
			Setting the size of the Mesh is scales the vertecies along the axes of the Meshes frame 
			of reference such that it fits into a Box of the given size.
		"""
		self.asset._prepare_for_modification(self)
		self.asset.size = size


	@property
	def scale(self) -> np.ndarray:
		"""
		Modifying this attribute will create a new :class:`blueprints.assets.MeshAsset` object if 
		the previous Asset was the children of multiple parents. Otherwise the Asset get modified 
		directly. 
		
		.. note::
			If :attr:`vertecies` should be changed directly such that it adheres to the given scale, 
			use :attr:`size` instead.
		
		Returns
		-------
		np.ndarray
			The scaling of X-axis, Y-axis and Z-axis in each component. Analogous to :attr:`size` without changing :attr:`vertecies`.
		"""
		return self.asset.scale


	@scale.setter
	@blue.restrict
	def scale(self, scale: np.ndarray|list[float|int]) -> None:
		"""
		Modifying this attribute will create a new :class:`blueprints.assets.MeshAsset` object if 
		the previous Asset was the children of multiple parents. Otherwise the Asset get modified 
		directly. 
		
		.. note::
			If :attr:`vertecies` should be changed directly such that it adheres to the given scale, 
			use :attr:`size` instead.
		
		Parameters
		----------
		scale : np.ndarray | list[float | int]
			The scaling of X-axis, Y-axis and Z-axis in each component. Analogous to :attr:`size` without changing :attr:`vertecies`.
		"""
		self.asset._prepare_for_modification(self)
		self.asset.scale = scale



class HField(blue.HFieldGeomType, BaseGeom):
	"""
	Height field are mesh-like Geoms with a rectangular shape and varing heights. 
	The height data can either be given as a file or as a 2D numpy array (alternatively 2D list). 
	Supported fileformats are PNG and the mujoco native HF format.
	
	.. code-block::
		:caption: Terrain Data
	
		>>> ...
		>>> spiral = np.array([ ... ])
		>>> hfield = blue.geoms.HField(terrain=spiral)
		# ALTERNATIVELY
		>>> hfield = blue.geoms.HField(filename='spiral.hf')
		>>> world.attach(hfield, plane, ball)

	.. image:: /_static/spiral.gif

	Terrain data can be edited at runtime by setting the hfield attribute.

	.. code-block::
		:caption: Runtime Modification

		>>> n_frames = 1000
		>>> terrain = blue.perlin((n_frames, 100, 100), periodic=True)
		>>> ... # FURTHER TERRAIN PROCESSING
		>>> hfield = blue.geoms.HField(terrain=terrain, color='orange')
		>>> world.attach(hfield, copy=False)
		>>> for n in range(total_steps//20):
		>>> 	    world.step(n_steps=20)
		>>> 	    hfield.terrain = terrain[n % n_frames, ...]

	.. image:: /_static/hfield_update.gif

	"""
	@blue.restrict
	def __init__(self, 
		     pos:           np.ndarray|list[int|float]      = [0., 0., 0.], 
		     terrain:       np.ndarray|list[int|float]|None = None, 
		     filename:      str|None                        = None, 
		     x_length:      int|float|None                  = 1, 
		     y_length:      int|float|None                  = 1, 
		     z_length:      int|float|None                  = 1, 
		     height_offset: int|float|None                  = 1, 
		     margin:        int|float   = 0.0,
		     gap:           int|float   = 0.0,
		     material:      blue.MaterialType|None          = None, 
		     asset:         blue.assets.HFieldAsset|None    = None, 
		     x:             int|float|np.int32|np.int64|np.float32|np.float64|None = None, 
		     y:             int|float|np.int32|np.int64|np.float32|np.float64|None = None, 
		     z:             int|float|np.int32|np.int64|np.float32|np.float64|None = None, 
		     **kwargs) -> None:
		"""
		Parameters
		----------
		pos : list[int | float] | np.ndarray | None, optional
			Represents the position of the Thing. Changing this attribute also changes the properties :attr:`x`, :attr:`y` and :attr:`z`.
		terrain : np.ndarray | list [ int | float ] | None
			The terrain data in the shape of a 2D array.
		filename : str | None, optional
			The name for the file from which the HField data is loaded. The supported formats are ``.png`` and ``.hf``.
		x_length : int | float, optional
			The length of the HField along the x-axis.
		y_length : int | float, optional
			The length of the HField along the y-axis.
		z_length : int | float, optional
			The length of the HField along the z-axis.
		height_offset : int | float | None
			The offset in height for the terrain data points.
		margin : int | float | None, optional
			A contact is considered active only if the distance between the two geom surfaces is below 
			margin-gap.
		gap : int | float | None, optional
			This attribute is used to enable the generation of inactive contacts, i.e., contacts that are 
			ignored by the constraint solver but are included in mjData.contact for the purpose of custom 
			computations. When this value is positive, geom distances between margin and margin-gap 
			correspond to such inactive contacts.
		material : blue.MaterialType, optional
			The :class:`Material <blueprints.material.Material>` of the Geom. 
			:class:`Textures <blueprints.texture.BaseTexture> are applied via Materials.
		asset : blue.assets.HFieldAsset | None, optional
			The Asset of the HField
		name : str | None, optional
			The user specified name might potentially be altered to avoid a naming conflict by appending an enumeration scheme.
		x : int | float |np.int32 | np.int64 | np.float32 | np.float64 | None, optional
			If `pos` is not specified, this argument sets the X position coordinate.
		y : int | float |np.int32 | np.int64 | np.float32 | np.float64 | None, optional
			If `pos` is not specified, this argument sets the Y position coordinate.
		z : int | float |np.int32 | np.int64 | np.float32 | np.float64 | None, optional
			If `pos` is not specified, this argument sets the Z position coordinate.
		**kwargs
			Keyword arguments passed to ``super().__init__``
		"""
		#self.terrain = terrain
		kwargs_nameless = kwargs.copy()
		if 'name' in kwargs_nameless:
			del kwargs_nameless['name']
		if sum(x is not None for x in (terrain, filename, asset)) > 1:
			raise Exception('Not more than one argument (terrain, file or asset) is allowed to be None.')
		if pos is None:
			x = float(x) if x is not None else 0.
			y = float(y) if y is not None else 0.
			z = float(z) if z is not None else 0.
			pos = np.array([x, y, z], dtype=np.float32)
		if asset is not None:
			self.asset = asset
		elif filename is not None:
			if not os.path.isfile(filename):
				# TRY DIRNAME PREFIX TO RESOLVE RELATIVE REF TO MAIN.PY
				filename = f'{os.path.dirname(sys.argv[0])}/{filename}'
				if not os.path.isfile(filename):
					raise Exception(f'File not found for path {filename}')
			if not os.path.isabs(filename):
				filename = os.path.abspath(filename)
			if not os.path.isabs(filename):
				path = os.path.abspath(os.path.dirname(sys.argv[0]))
				filename = f'{path}/{filename}'
			self.asset = blue.assets.HFieldAsset(filename=filename, 
							     pos=pos, 
							     x_length=x_length, 
							     y_length=y_length, 
							     z_length=z_length, 
							     height_offset=height_offset, 
							     xml_data=False, 
							     **kwargs_nameless)
		elif terrain is not None:
			self.asset = blue.assets.HFieldAsset(terrain=terrain, 
							     pos=pos, 
							     x_length=x_length, 
							     y_length=y_length, 
							     z_length=z_length, 
							     height_offset=height_offset, 
							     xml_data=True, 
							     **kwargs_nameless)
		else:
			raise Exception('No asset could be constructed! Please provide either a filename, terrain or an asset.')
		self.asset._add(self)
		self.asset.freeze = True
		super().__init__(pos=pos,
				 material=material, 
				 margin=margin, 
				 gap=gap, 
				 **kwargs)
		self.asset.freeze = False


	def __getitem__(self, 
			key: tuple[slice|int]) -> np.ndarray|np.float32:
		"""
		Returns
		-------
		np.ndarray | np.float32
			The values from the selected index/slice of the height field.
		"""
		return self.asset[key]


	def __setitem__(self, 
			key:   tuple[slice|int], 
			value: int|float|list[int|float]|np.ndarray) -> None:
		"""
		Parameters
		----------
		key : tuple[slice | int]
			The indecies/slices of acces
		value : int|float|list[int|float]|np.ndarray
			The value to be assigned in the selected parts of the field.
		"""
		self.asset._prepare_for_modification(self)
		self.asset[key] = value


	@blue.restrict
	def _build(self, 
		   parent, 
		   world, 
		   indicies, 
		   **kwargs):
		"""
		This method is called to build the xml.
		
		Parameters
		----------
		parent : xml.etree.ElementTree.Element
			The xml element of its parent
		world : WorldType
			The World from which the build method was called initially
		
		Returns
		-------
		xml.etree.ElementTree.Element
			The builded xml element of the Thing.
		"""
		if self.material is not None:
			self.material._build(parent, world, indicies, **kwargs)
			kwargs['material'] = self.material.asset.name
		self._xml_root = xml.SubElement(parent, 
						self._MUJOCO_OBJ, 
						hfield=self.asset.name, 
						**self._mujoco_specs(kwargs))
		if self.asset._built:
			self._index = self.asset._index
		else:
			self._index         = indicies['hfield']
			indicies['hfield'] += 1
		self.asset._build(parent=parent, 
				  world=world, 
				  indicies=indicies, 
				  **kwargs)
		return self._xml_root

	# MUJOCO PROPERTIES

	@property
	def pos(self) -> np.ndarray:
		"""
		Modifying this attribute will create a new :class:`blueprints.assets.MeshAsset` object if 
		the previous Asset was the children of multiple parents. Otherwise the Asset get modified 
		directly. 
		
		Returns
		-------
		np.ndarray | list[float | int]
		"""
		return self.asset.pos.copy()


	@pos.setter
	@blue.restrict
	def pos(self, pos: np.ndarray|list[float|int]) -> None:
		"""
		Modifying this attribute will create a new :class:`blueprints.assets.MeshAsset` object if 
		the previous Asset was the children of multiple parents. Otherwise the Asset get modified 
		directly. 
		
		Parameters
		----------
		pos : np.ndarray | list[float | int]
			Description
		
		Deleted Parameters
		------------------
		vertex : list[np.ndarray | list[float | int]]
		Description
		"""
		self.asset._prepare_for_modification(self)
		self.asset.pos = pos


	@property
	def terrain(self) -> np.ndarray:
		"""
		Modifying this attribute will create a new :class:`blueprints.assets.MeshAsset` object if 
		the previous Asset was the children of multiple parents. Otherwise the Asset get modified 
		directly. 
		
		Returns
		-------
		np.ndarray | list[float | int]
		"""
		return self.asset.terrain.copy()


	@terrain.setter
	@blue.restrict
	def terrain(self, terrain: np.ndarray|list[float|int]) -> None:
		"""
		Modifying this attribute will create a new :class:`blueprints.assets.MeshAsset` object if 
		the previous Asset was the children of multiple parents. Otherwise the Asset get modified 
		directly. 
		
		Parameters
		----------
		terrain : np.ndarray | list[float | int]
			Description
		
		Deleted Parameters
		------------------
		vertex : list[np.ndarray | list[float | int]]
		Description
		"""
		#self.asset._prepare_for_modification(self)
		self.asset.terrain = terrain


	@property
	def euler(self) -> np.ndarray:
		"""
		Modifying this attribute will create a new :class:`blueprints.assets.MeshAsset` object if 
		the previous Asset was the children of multiple parents. Otherwise the Asset get modified 
		directly. 
		
		Returns
		-------
		np.ndarray
		"""
		return self.asset.euler.copy()


	@euler.setter
	@blue.restrict
	def euler(self, euler: np.ndarray|list[float|int]) -> None:
		"""
		Modifying this attribute will create a new :class:`blueprints.assets.MeshAsset` object if 
		the previous Asset was the children of multiple parents. Otherwise the Asset get modified 
		directly. 
		
		Parameters
		----------
		euler : np.ndarray | list[float | int]
			Description
		
		Deleted Parameters
		------------------
		vertex : list[np.ndarray | list[float | int]]
		Description
		pos : np.ndarray | list[float | int]
		Description
		"""
		self.asset._prepare_for_modification(self)
		self.asset.euler = euler


	@property
	def size(self) -> np.ndarray:
		"""
		Modifying this attribute will create a new :class:`blueprints.cache.MeshCache` object if 
		the previous Cache was the children of multiple parents. Otherwise the Cache get modified 
		directly. 
		
		.. note::
			Altering this attribute also changes the :attr:`vertecies` attribute. To get the same 
			effect without changing the vertecies, see :attr:`scale`.
		
		Returns
		-------
		np.ndarray
			The size of the Mesh is calculated as the size of the Box with edges along the axes of 
			the Meshes frame of reference, that captures all vertecies.
		"""
		return self.asset.size.copy()


	@size.setter
	@blue.restrict
	def size(self, size: np.ndarray|list[float|int]) -> None:
		"""
		Modifying this attribute will create a new :class:`blueprints.cache.MeshCache` object if 
		the previous Cache was the children of multiple parents. Otherwise the Cache get modified 
		directly.

		.. note::
			Altering this attribute also changes the :attr:`vertecies` attribute. To get the same 
			effect without changing the vertecies, see :attr:`scale`.
		
		Parameters
		----------
		size : np.ndarray | list[float | int]
			Setting the size of the Mesh is scales the vertecies along the axes of the Meshes frame 
			of reference such that it fits into a Box of the given size.
		"""
		self.asset._prepare_for_modification(self)
		self.asset.size = size

	# BLUEPRINTS ATTRIBUTES


	@property
	def x_length(self) -> float:
		"""
		Length along the ``X``-axis of the HField.

		Returns
		-------
		float
		"""
		return self.asset.x_length


	@property
	def y_length(self) -> float:
		"""
		Length along the ``Y``-axis of the HField.

		Returns
		-------
		float
		"""
		return self.asset.y_length


	@property
	def z_length(self) -> float:
		"""
		Length along the ``Z``-axis of the HField.

		Returns
		-------
		float
		"""
		return self.asset.z_length


	@property
	def height_offset(self):
		"""
		Height offset along the ``Z``-axis of the HField.

		Returns
		-------
		float
		"""
		return self.asset.height_offset


	@x_length.setter
	@blue.restrict
	def x_length(self, x_length: int|float|np.int32|np.int64|np.float32|np.float64) -> None:
		self.asset.x_length = float(x_length)


	@y_length.setter
	@blue.restrict
	def y_length(self, y_length: int|float|np.int32|np.int64|np.float32|np.float64) -> None:
		self.asset.y_length = float(y_length)


	@z_length.setter
	@blue.restrict
	def z_length(self, z_length: int|float|np.int32|np.int64|np.float32|np.float64) -> None:
		self.asset.z_length = float(z_length)


	@height_offset.setter
	@blue.restrict
	def height_offset(self, height_offset: int|float|np.int32|np.int64|np.float32|np.float64) -> None:
		self.asset.height_offset = float(height_offset)
	




GEOM_THINGS = {'capsule':   Capsule, 
	       'cylinder':  Cylinder, 
	       'plane':     Plane, 
	       'sphere':    Sphere, 
	       'ellipsoid': Ellipsoid, 
	       'box':       Box, 
	       'mesh':      Mesh, 
	       'hfield':    HField}
