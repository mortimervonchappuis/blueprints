import numpy as np
import xml.etree.ElementTree as xml
import blueprints as blue



class BaseSite(blue.SiteType, blue.thing.MoveableThing, blue.thing.ColoredThing, blue.thing.NodeThing):

	"""
	Sites are used to make a models sensitive. If a :class:`blueprints.sensors.BaseSensor` is attached 
	to the Site, it becomes sensitive to collisions of the Site with other physical Things. To do this 
	Sites are not physical Things themselves in the sense, that they do not posses mass. For different 
	Site shapes individual Site classes are available, with shape depended size attributes. All other 
	attributes are specified in :class:`BaseSite`.
	
	.. note::
		Though units are not bound to one specific system of measurements, it is highly recommended 
		to use `MKS <https://en.wikipedia.org/wiki/MKS_system_of_units>`__, since all defaults are 
		defined based on meters, kilograms and seconds and most mujoco models online also use the 
		metric system. If you want to switch from the imperial to the metric system the following resource might be 
		of interest to `convert units from imperial to metric <https://xkcd.com/526/>`__. If you insist on 
		using imperial units, take a look at the following resource to `convert units from metric to imperial <https://rick.nerial.uk/>`__.
	"""

	@blue.restrict
	def __init__(self, 
		     pos:       np.ndarray|list[int|float]                             = [0., 0., 0.], 
		     alpha:     int|float                                              =  0, 
		     beta:      int|float                                              =  0, 
		     gamma:     int|float                                              =  0, 
		     material:  blue.MaterialType|None                                 =  None, 
		     color:     object|None                                            =  None, 
		     sensors:   blue.SensorType|list[blue.SensorType]|None             =  None, 
		     actuators: blue.ActuatorType|list[blue.ActuatorType]|None         =  None, 
		     name:      str|None                                               =  None, 
		     copy:      bool                                                   =  True, 
		     x:         int|float|np.int32|np.int64|np.float32|np.float64|None =  None, 
		     y:         int|float|np.int32|np.int64|np.float32|np.float64|None =  None, 
		     z:         int|float|np.int32|np.int64|np.float32|np.float64|None =  None, 
		     quat:      list[int|float]|np.ndarray|None                        =  None, 
		     **kwargs):
		"""
		Parameters
		----------
		pos : np.ndarray | list[int | float], optional
			Represents the position of the object. Changing this attribute also changes the properties :attr:`x`, :attr:`y` and :attr:`z`.
		alpha : int | float, optional
			(Improper) euler angle of rotation around the x-axis in radian. Changing this value also changes the :attr:`euler` property.
		beta : int | float, optional
			(Improper) euler angle of rotation around the y-axis in radian. Changing this value also changes the :attr:`euler` property.
		gamma : int | float, optional
			(Improper) euler angle of rotation around the z-axis in radian. Changing this value also changes the :attr:`euler` property.
		color : blue.ColorType
			The color of the Site. For a detailed description see :class:`Color `
		sensors : blue.SensorType | list[blue.SensorType] | None, optional
			The ``sensors`` argument mus either be a list of :class:`sensors.BaseSensor` or a single :class:`sensors.BaseSensor`.
		actuators : blue.ActuatorType | list[blue.ActuatorType] | None, optional
			The ``actuators`` argument mus either be a list of :class:`actuators.BaseActuator` or a single :class:`actuators.BaseActuator`.
		name : str | None, optional
			The user specified name of the Site. In the case of a naming conflict the name is appended by an enumeration scheme.
		copy : bool, optional
			This argument indicates whether the children should be copied before they are attached. 
			If copy if set to False this will result in graph cycles if a parent of the Body is 
			attached resulting in further errors.
		x : int | float |np.int32 | np.int64 | np.float32 | np.float64 | None, optional
			If `pos` is not specified, this argument sets the X position coordinate.
		y : int | float |np.int32 | np.int64 | np.float32 | np.float64 | None, optional
			If `pos` is not specified, this argument sets the Y position coordinate.
		z : int | float |np.int32 | np.int64 | np.float32 | np.float64 | None, optional
			If `pos` is not specified, this argument sets the Z position coordinate.
		quat: list [ int | float ] | np.ndarray | None, optional
			If set, the quaternion orientation overwrites the euler angles ``alpha``, ``beta`` and ``gamma``.
		**kwargs
			Description
		"""
		pack = lambda x: x if isinstance(x, list) else [x]
		sensors              = pack(sensors)   if sensors   else []
		actuators            = pack(actuators) if actuators else []
		self._sensors        = []
		self._actuators      = []
		self._ref_actuators  = []
		self._tendons        = []
		self._side_tendons   = []
		self._CHILDREN  = {'sensors':        {'type':     blue.SensorType, 
						      'children': self._sensors}, 
				   'actuators':      {'type':     blue.ActuatorType, 
						      'children': self._actuators}, 
				   'tendons':        {'type':     blue.TendonType, 
						      'children': self._tendons}, 
				   'side_tendons':   {'type':     blue.TendonType, 
						      'children': self._side_tendons}}
		self._CYCLE_REF = {'ref_actuators':  {'type':     blue.ActuatorType, 
						      'children': self._ref_actuators}}
		# SETTING COLOR BLUEPRINT COLOR DEFAULT FOR SITES
		color = color if color is not None else 'orange'
		self.material = material
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
		self.attach(*sensors, 
			    *actuators, 
			    copy=copy)
		self._check_children_types()


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
		self._index       = indicies['site']
		indicies['site'] += 1
		return super()._build(parent, world, indicies, **kwargs)

	# KINEMATIC TREES PROPERTIES/METHODS

	@property
	def ref_actuators(self) -> blue.ViewType:
		"""
		Ref Actuators contain :class:`blueprints.actuators.BaseActuator` that are attach to another 
		:class:`Site` to which they apply force, using this Sites reference frame.
		
		Returns
		-------
		:class:`blueprints.utils.view.View`
			A View of the Actuators that use the reference frame of this Site to apply force to 
			another Site.
		"""
		return blue.View(self._ref_actuators, name='ref_actuators', parent=self)


	@ref_actuators.setter
	@blue.restrict
	def ref_actuators(self, actuators: blue.ActuatorType|list[blue.ActuatorType]|blue.ViewType):
		"""
		Ref Actuators contain :class:`blueprints.actuators.BaseActuator` that are attach to another 
		:class:`Site` to which they apply force, using this Sites reference frame.
		
		Parameters
		----------
		actuators : list[blue.ActuatorType] | blue.ViewType
			A list of the Actuators that use the reference frame of this Site to apply force to 
			another Site.
		"""
		actuators = [actuators] if isinstance(actuators, blue.ActuatorType) else actuators
		for actuator in actuators:
			actuator.refsite = self

	# XML PROPERTIES/METHODS

	@blue.restrict
	@classmethod
	def _from_xml_element(cls, 
			      xml_element:   xml.Element, 
			      sensors:       list = [], 
			      actuators:     list = [], 
			      ref_actuators: list = []) -> blue.ThingType:
		"""
		This method reconstructs a Site from an xml element.
		
		Parameters
		----------
		xml_element : xml.Element
			The xml element from which a Body is reconstructed.
		sensors : list, optional
			A list of Sensors, that are attached to the resulting Site.
		actuators : list, optional
			A list of Actuators, that are attached to the resulting Site.
		ref_actuators : list, optional
			A list of Ref Actuators, that are attached to the resulting Site.
		
		Returns
		-------
		blue.ThingType
			The reconstructed Site.
		"""
		init_args, post_args, rest_args = cls._xml_element_args(xml_element)
		init_args['sensors'] = sensors
		init_args['copy']    = False
		if 'type' in rest_args:
			site_type = rest_args.pop('type')
		else:
			site_type = None
		site = object.__new__(blue.REGISTER.SITE_THINGS[site_type])
		site.__init__(**init_args)
		for key, val in post_args.items():
			setattr(site, key, val)
		for actuator in actuators:
			actuator.site = site
		for actuator in ref_actuators:
			actuator.refsite = site
		return site

	# MUJOCO PROPERTIES

	@property
	def type(self) -> str:
		"""
		This is a derived attribute, which is used to specify the ``type`` attribute in the 
		mujoco ``site`` tag.

		Returns
		-------
		str
			The type is the lower case name of the Site class.
		"""
		return self.__class__.__name__.lower()


	@property
	def material(self):
		"""
		Materials can be used to specify reflective properties of a Site, 
		as well as to apply :class:`Textures <blueprints.texture.BaseTexture>` to them.

		Returns
		-------
		blue.MaterialType
			The Material of the Site
		"""
		return self._material


	@material.setter
	@blue.restrict
	def material(self, material: blue.MaterialType|None) -> None:
		self._material = material.copy() if material is not None else material



class Capsule(blue.CapsuleSiteType, BaseSite, blue.tube.BaseTube):

	"""
	Capsules consist of a cylinder with two half spheres at its ends. It can either be constructed from 
	its standard constructor for position, orientation, radius and length, or with the :meth:`blueprints.tube.BaseTube` 
	constructor. For a detailed description see :class:`blueprints.tube.BaseTube`.

	.. code-block:: python
		:caption: Standard constructor:

		>>> site = blue.sites.Capsule(alpha=TAU/8, length=3)
		>>> site.head
		array([ 0.          1.06066017 -1.06066017], dtype=np.float32)
		>>> site.tail
		array([ 0.         -1.06066017  1.06066017], dtype=np.float32)

	.. code-block:: python
		:caption: From points constructor:

		>>> site = blue.sites.Capsule.from_points([0, -1, 4], [2, 1, 0], radius=0.5)
		>>> site.pos
		array([1. 0. 2.], dtype=np.float32)
		>>> site.alpha, site.beta, site.gamma
		-2.6779450445889874 0.420534335283965 0.6847192030022825
		>>> site.length
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
		     radius:   int|float   = 1, 
		     length:   int|float   = 1, 
		     alpha:    int|float   = 0.,
		     beta:     int|float   = 0.,
		     gamma:    int|float   = 0., 
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
		color : blue.ColorType
			The color of the Site. For a detailed description see :class:`Color `
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
				 self.length/2,
				 0], dtype=np.float64)


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



class Cylinder(blue.CylinderSiteType, BaseSite, blue.tube.BaseTube):

	"""
	Cylinders can either be constructed from their standard constructor for position, orientation, 
	radius and length, or with the :meth:`blueprints.tube.BaseTube` constructor. For a detailed description 
	see :class:`blueprints.tube.BaseTube`.

	.. code-block:: python
		:caption: Standard constructor:

		>>> site = blue.sites.Cylinder(alpha=TAU/8, length=3)
		>>> site.head
		array([ 0.          1.06066017 -1.06066017], dtype=np.float32)
		>>> site.tail
		array([ 0.         -1.06066017  1.06066017], dtype=np.float32)

	.. code-block:: python
		:caption: From points constructor:

		>>> site = blue.sites.Cylinder.from_points([0, -1, 4], [2, 1, 0], radius=0.5)
		>>> site.pos
		array([1. 0. 2.], dtype=np.float32)
		>>> site.alpha, site.beta, site.gamma
		-2.6779450445889874 0.420534335283965 0.6847192030022825
		>>> site.length
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
		     radius:   int|float = 1, 
		     length:   int|float = 1, 
		     alpha:    int|float = 0.,
		     beta:     int|float = 0.,
		     gamma:    int|float = 0., 
		     material: blue.MaterialType|None = None, 
		     color:    object|None = None, 
		     name:     str|None  = None, 
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
		color : blue.ColorType
			The color of the Site. For a detailed description see :class:`Color `
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
				 self.length/2,
				 0], dtype=np.float64)


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



class Box(blue.BoxSiteType, BaseSite):

	"""
	Boxes can either be constructed from their standard constructor for position, orientation, 
	and x-, y-, z-length, or with the :meth:`blueprints.tube.BaseTube` constructor. For a detailed 
	description see :class:`blueprints.tube.BaseTube`.

	.. code-block:: python
		:caption: Standard constructor:

		>>> site = blue.sites.Box(x_length=2, y_length=4, z_length=1)
		>>> site.head
		array([ 0.   0.  -0.5], dtype=np.float32)
		>>> site.tail
		array([ 0.   0.   0.5], dtype=np.float32)

	.. code-block:: python
		:caption: From points constructor:

		>>> site = blue.sites.Box.from_points([4, 0,-2], [2, 4, 0], radius=0.5)
		>>> site.pos
		array([ 3.   2.  -1.], dtype=np.float32)
		>>> site.alpha, site.beta, site.gamma
		-1.1071487177940904 -0.4205343352839653 0.20135792079032988
		>>> site.x_length, site.y_length, site.z_length
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
		     size:     np.ndarray|list[int|float]|int|float|None = None, 
		     x_length: int|float   = 1, 
		     y_length: int|float   = 1, 
		     z_length: int|float   = 1, 
		     alpha:    int|float   = 0.,
		     beta:     int|float   = 0.,
		     gamma:    int|float   = 0., 
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
		color : blue.ColorType
			The color of the Site. For a detailed description see :class:`Color `
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
		if size is not None:
			if isinstance(size, list):
				if len(size) == 3:
					x_length = size[0]
					y_length = size[1]
					z_length = size[2]
				else:
					raise ValueError(f'If size is a list it must have a length of 3, got {size.shape} instead.')
			elif isinstance(size, np.ndarray):
				if size.shape == (3,):
					x_length = size[0]
					y_length = size[1]
					z_length = size[2]
				else:
					raise ValueError(f'If size is a numpy.ndarray it must be of shape (3,), got {size.shape} instead.')
			elif isinstance(size, (int, float)):
				x_length = size
				y_length = size
				z_length = size
		self.x_length = float(x_length)
		self.y_length = float(y_length)
		self.z_length = float(z_length)
		super().__init__(pos=pos, 
				 alpha=alpha, 
				 beta=beta, 
				 gamma=gamma, 
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
				 self.z_length/2], dtype=np.float64)


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



class Sphere(blue.SphereSiteType, BaseSite):

	"""
	Spheres are defined via radius and position.

	Attributes
	----------
	radius : float
		The radius of the sphere
	"""
	
	@blue.restrict
	def __init__(self, 
		     pos:      np.ndarray|list[int|float] = [0., 0., 0.],
		     radius:   int|float   = 1, 
		     alpha:    int|float   = 0.,
		     beta:     int|float   = 0.,
		     gamma:    int|float   = 0., 
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
			The radius of the Sphere.
		alpha : int | float | None, optional
			(Improper) euler angle of rotation around the x-axis in radian. Changing this value also changes the :attr:`euler` property.
		beta : int | float | None, optional
			(Improper) euler angle of rotation around the y-axis in radian. Changing this value also changes the :attr:`euler` property.
		gamma : int | float | None, optional
			(Improper) euler angle of rotation around the z-axis in radian. Changing this value also changes the :attr:`euler` property.
		color : blue.ColorType
			The color of the Site. For a detailed description see :class:`Color `
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
		return np.array([self.radius,
				 0,
				 0], dtype=np.float64)


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



class Ellipsoid(blue.EllipsoidSiteType, BaseSite):

	"""
	Ellipsoids are defined with via :attr:`x_radius`, :attr:`y_radius` and :attr:`z_radius`. The surface of 
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
		     x_radius: int|float   = 1,
		     y_radius: int|float   = 1,
		     z_radius: int|float   = 1, 
		     alpha:    int|float   = 0.,
		     beta:     int|float   = 0.,
		     gamma:    int|float   = 0., 
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
		color : blue.ColorType
			The color of the Site. For a detailed description see :class:`Color `
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
				 self.z_radius], dtype=np.float64)


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



class Site(Sphere):

	"""
	Dummy class for :class:`Sphere`
	"""

	@property
	def type(self):
		"""
		The type is the lower case name of the Sites class, in this case ``'sphere'``.
		
		Returns
		-------
		str
		"""
		return 'sphere'



SITE_THINGS = {None:        Site, 
	       'capsule':   Capsule, 
	       'cylinder':  Cylinder, 
	       'sphere':    Sphere, 
	       'ellipsoid': Ellipsoid, 
	       'box':       Box}
