import numpy as np
import xml.etree.ElementTree as xml
import blueprints as blue



class BaseJoint(blue.JointType, blue.MoveableThing, blue.thing.NodeThing):

	"""
	The Joint class represents Mujoco joint objects. All Joints inherit from this base class.

	.. note::
		Joints enable movement along the degrees of freedom their parent Bodies and not to Bodies 
		attached to the Joints (one cannot attach Bodies to Joints). This might be counterintuitive 
		since real joints move objects which are attached to them.

	Most attribute descriptions are partially taken from `Mujoco <https://mujoco.readthedocs.io/en/latest/XMLreference.html#body-joint>`__.
	"""

	@blue.restrict
	def __init__(self, 
		     pos:                np.ndarray|list[int|float] = [0., 0., 0.], 
		     alpha:              int|float = 0., 
		     beta:               int|float = 0., 
		     gamma:              int|float = 0., 
		     springdamper:       np.ndarray|list[int|float] = [0., 0.], 
		     actuatorforcerange: np.ndarray|list[int|float] = [0., 0.], 
		     stiffness:          int|float = 0., 
		     springref:          int|float = 0., 
		     armature:           int|float = 0., 
		     damping:            int|float = 0., 
		     sensors:            blue.SensorType|list[blue.SensorType]|None     = None, 
		     actuators:          blue.ActuatorType|list[blue.ActuatorType]|None = None, 
		     name:               str|None  = None, 
		     copy:               bool      = True, 
		     x:                  int|float|np.int32|np.int64|np.float32|np.float64|None = None, 
		     y:                  int|float|np.int32|np.int64|np.float32|np.float64|None = None, 
		     z:                  int|float|np.int32|np.int64|np.float32|np.float64|None = None, 
		     **kwargs):
		"""
		Parameters
		----------
		pos : np.ndarray | list[int | float], optional
			Represents the position of the object. Changing this attribute also changes the properties 
			:attr:`x`, :attr:`y` and :attr:`z`.
		alpha : int | float, optional
			(Improper) euler angle of rotation around the x-axis in radian. Changing this value also 
			changes the :attr:`euler` property.
		beta : int | float, optional
			(Improper) euler angle of rotation around the y-axis in radian. Changing this value also 
			changes the :attr:`euler` property.
		gamma : int | float, optional
			(Improper) euler angle of rotation around the z-axis in radian. Changing this value also 
			changes the :attr:`euler` property.
		springdamper : np.ndarray | list[int | float], optional
			When both numbers are positive, the compiler will override any stiffness and damping 
			values specified with the attributes below, and will instead set them automatically so 
			that the resulting mass-spring-damper for this joint has the desired time constant 
			(first value) and damping ratio (second value). 
		actuatorforcerange : np.ndarray | list[int | float], optional
			This attribute specifies whether actuator forces acting on the Joint should be clamped.
		stiffness : int | float, optional
			Joint stiffness. If this value is positive, a spring will be created with equilibrium 
			position given by springref below. The spring force is computed along with the other 
			passive forces.
		springref : int | float, optional
			The Joint position or angle in which the joint spring (if any) achieves equilibrium.
		armature : int | float, optional
			Armature inertia (or rotor inertia, or reflected inertia) of all degrees of freedom 
			created by this Joint. These are constants added to the diagonal of the inertia matrix 
			in generalized coordinates. They make the simulation more stable, and often increase 
			physical realism. This is because when a motor is attached to the system with a transmission 
			that amplifies the motor force by :math:`c`, the inertia of the rotor (i.e., the moving part 
			of the motor) is amplified by :math:`c\\cdot c`. The same holds for gears in the early 
			stages of planetary gear boxes. These extra inertias often dominate the inertias of the 
			robot parts that are represented explicitly in the model, and the armature attribute is the 
			way to model them.
		damping : int | float, optional
			Damping applied to all degrees of freedom created by this joint. Unlike friction loss which 
			is computed by the constraint solver, damping is simply a force linear in velocity. It is 
			included in the passive forces. Despite this simplicity, larger damping values can make 
			numerical integrators unstable, which is why our Euler integrator handles damping implicitly.
		sensors : blue.SensorType | list[blue.SensorType] | None, optional
			The Sensors are attached to the actuator on initialization.
		actuators : blue.ActuatorType | list[blue.ActuatorType] | None, optional
			The Actuators are attached to the actuator on initialization.
		name : str | None, optional
			The user specified name. In the case of a naming conflict the name will be altered by an 
			enumeration scheme.
		copy : bool, optional
			If True, the Sensors and Actuators passed during init will be copied before attachment, 
			otherwise the original Sensor is attached.
		**kwargs
			Keyword arguments are passed to ``super().__init__``.
		"""
		pack      = lambda x: x if isinstance(x, list) else [x]
		sensors   = pack(sensors)   if sensors   else []
		actuators = pack(actuators) if actuators else []
		self._sensors           = []
		self._tendons           = []
		self._actuators         = []
		self._CHILDREN          = {'tendons':   {'type':     blue.TendonType, 
							 'children': self._tendons}, 
					   'sensors':   {'type':     blue.SensorType, 
							 'children': self._sensors}, 
					   'actuators': {'type':     blue.ActuatorType, 
							 'children': self._actuators}}
		self.springdamper       = springdamper
		self.actuatorforcerange = actuatorforcerange
		self.stiffness          = stiffness
		self.springref          = springref
		self.armature           = armature
		self.damping            = damping
		super().__init__(pos=pos, 
				 alpha=alpha, 
				 beta=beta, 
				 gamma=gamma, 
				 name=name, 
				 x=x, 
				 y=y, 
				 z=z, 
				 **kwargs)
		self.attach(*sensors, 
			    *actuators, 
			    copy=copy)
		for actuator in self.actuators:
			actuator.joint = self
		self._check_children_types()


	@blue.restrict
	def _build(self, 
		   parent: xml.Element, 
		   world, 
		   indicies, 
		   **kwargs) -> xml.Element:
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
			The builded xml element of the Joint.
		"""
		self._index      = indicies['jnt']
		indicies['jnt'] += 1
		if hasattr(self, 'axis'):
			if 'axis' in kwargs:
				axis = kwargs.pop('axis')
			else:
				axis = self.axis
			kwargs['axis'] = self.rotation_matrix @ axis
		return super()._build(parent, world, indicies, **kwargs)


	@blue.restrict
	@classmethod
	def _from_xml_element(cls, 
				  xml_element: xml.Element, 
				  sensors:     list = None, 
				  actuators:   list = []) -> blue.ThingType:
		"""
		This method reconstructs a Joint from an xml element.
		
		Parameters
		----------
		xml_element : xml.Element
			The xml element from which a Geom is reconstructed.
		sensors : blue.SensorType | list[blue.SensorType] | None, optional
			The Sensors are attached to the actuator on initialization.
		actuators : blue.ActuatorType | list[blue.ActuatorType] | None, optional
			The Actuators are attached to the actuator on initialization.
		
		Returns
		-------
		blue.ThingType
			The reconstructed Joint.
		"""
		init_args, post_args, rest_args = cls._xml_element_args(xml_element)
		init_args['sensors'] = sensors
		init_args['copy']    = False
		if 'type' in rest_args:
			joint_type = rest_args.pop('type')
		else:
			joint_type = None
		joint = object.__new__(blue.REGISTER.JOINT_THINGS[joint_type])
		joint.__init__(**init_args)
		for key, val in post_args.items():
			setattr(joint, key, val)
		for actuator in actuators:
			actuator.joint = joint
		return joint

	# MUJOCO PROPERTIES

	@property
	def type(self) -> str:
		"""
		This is a derived attribute, which is used to specify the ``type`` attribute in the 
		mujoco ``joint`` tag.

		Returns
		-------
		str
			The type is the lower case name of the Joint class.
		"""
		return self.__class__.__name__.lower()


	@property
	def springdamper(self) -> np.ndarray:
		"""
		When both numbers are positive, the compiler will override any stiffness and damping 
		values specified with the attributes below, and will instead set them automatically so 
		that the resulting mass-spring-damper for this joint has the desired time constant 
		(first value) and damping ratio (second value). 
		
		Returns
		-------
		np.ndarray
		"""
		return self._springdamper


	@springdamper.setter
	@blue.restrict
	def springdamper(self, springdamper: np.ndarray|list[int|float]) -> None:
		"""
		When both numbers are positive, the compiler will override any stiffness and damping 
		values specified with the attributes below, and will instead set them automatically so 
		that the resulting mass-spring-damper for this joint has the desired time constant 
		(first value) and damping ratio (second value). 
		
		Parameters
		----------
		springdamper : np.ndarray | list[int | float]
			The springdamer which is assigned.
		"""
		if isinstance(springdamper, list):
			if len(springdamper) != 2:
				raise ValueError(f'springdamper must have a length of 2, got {len(springdamper)} instead.')
		elif isinstance(springdamper, np.ndarray):
			if springdamper.shape != (2,):
				raise ValueError(f'springdamper must have a shape of (2,), got {springdamper.shape} instead.')
		self._springdamper = np.array(springdamper, dtype=np.float32)


	@property
	def actuatorforcerange(self) -> np.ndarray:
		"""
		actuatorforcerange : np.ndarray | list[int | float], optional
		This attribute specifies whether actuator forces acting on the Joint should be clamped.
		
		Returns
		-------
		np.ndarray
		"""
		return self._actuatorforcerange


	@actuatorforcerange.setter
	@blue.restrict
	def actuatorforcerange(self, actuatorforcerange: np.ndarray|list[int|float]) -> None:
		"""
		This attribute specifies whether actuator forces acting on the Joint should be clamped.
		
		Parameters
		----------
		actuatorforcerange : np.ndarray | list[int | float], optional
			The actuator force range which is assigned.
		"""
		if isinstance(actuatorforcerange, list):
			if len(actuatorforcerange) != 2:
				raise ValueError(f'actuatorforcerange must have a length of 2, got {len(actuatorforcerange)} instead.')
		elif isinstance(actuatorforcerange, np.ndarray):
			if actuatorforcerange.shape != (2,):
				raise ValueError(f'actuatorforcerange must have a shape of (2,), got {actuatorforcerange.shape} instead.')
		self._actuatorforcerange = np.array(actuatorforcerange, dtype=np.float32)


	@property
	def stiffness(self) -> float:
		"""
		Joint stiffness. If this value is positive, a spring will be created with equilibrium 
		position given by springref below. The spring force is computed along with the other 
		passive forces.
		
		Returns
		-------
		float
		"""
		return self._stiffness


	@stiffness.setter
	@blue.restrict
	def stiffness(self, stiffness: int|float) -> None:
		"""
		Joint stiffness. If this value is positive, a spring will be created with equilibrium 
		position given by springref below. The spring force is computed along with the other 
		passive forces.
		
		Parameters
		----------
		stiffness : int | float, optional
			The stiffness which is assigned
		"""
		self._stiffness = float(stiffness)


	@property
	def springref(self) -> float:
		"""
		The Joint position or angle in which the joint spring (if any) achieves equilibrium.
		
		Returns
		-------
		float
		"""
		return self._springref


	@springref.setter
	@blue.restrict
	def springref(self, springref: int|float) -> None:
		"""
		The Joint position or angle in which the joint spring (if any) achieves equilibrium.
		
		Parameters
		----------
		springref : int | float, optional
			The springref which is assigned
		"""
		self._springref = float(springref)


	@property
	def armature(self) -> float:
		"""
		Armature inertia (or rotor inertia, or reflected inertia) of all degrees of freedom 
		created by this Joint.
		
		Returns
		-------
		float
		"""
		return self._armature


	@armature.setter
	@blue.restrict
	def armature(self, armature: int|float) -> None:
		"""
		Armature inertia (or rotor inertia, or reflected inertia) of all degrees of freedom created 
		by this Joint.
		
		Parameters
		----------
		armature : int | float
			The armature which is assigned
		"""
		self._armature = float(armature)


	@property
	def damping(self) -> float:
		"""
		Damping applied to all degrees of freedom created by this joint. Unlike friction loss which 
		is computed by the constraint solver, damping is simply a force linear in velocity. It is 
		included in the passive forces. Despite this simplicity, larger damping values can make 
		numerical integrators unstable, which is why our Euler integrator handles damping implicitly.
		
		Returns
		-------
		float
		"""
		return self._damping


	@damping.setter
	@blue.restrict
	def damping(self, damping: int|float) -> None:
		"""
		Damping applied to all degrees of freedom created by this joint. Unlike friction loss which is 
		computed by the constraint solver, damping is simply a force linear in velocity. It is included 
		in the passive forces. Despite this simplicity, larger damping values can make numerical 
		integrators unstable, which is why our Euler integrator handles damping implicitly.
		
		Parameters
		----------
		damping : int | float
			The damping which is assigned
		"""
		self._damping = float(damping)



class Hinge(blue.HingeType, BaseJoint):

	"""
	Most attribute descriptions are partially taken from `Mujoco <https://mujoco.readthedocs.io/en/latest/XMLreference.html#body-joint>`__.
	"""

	@blue.restrict
	def __init__(self, 
		     pos:          np.ndarray|list[int|float]     = [0., 0., 0.], 
		     axis:         np.ndarray|list[int|float]|str = [0., 0., 1.], 
		     range:        np.ndarray|list[int|float]     = [0., 0.], 
		     ref:          int|float = 0., 
		     frictionloss: int|float = 0., 
		     name:         str|None  = None, 
		     x:            int|float|np.int32|np.int64|np.float32|np.float64|None = None, 
		     y:            int|float|np.int32|np.int64|np.float32|np.float64|None = None, 
		     z:            int|float|np.int32|np.int64|np.float32|np.float64|None = None, 
		     **kwargs):
		"""
		Parameters
		----------
		pos : np.ndarray | list[int | float], optional
			Represents the position of the object. Changing this attribute also changes the properties 
			:attr:`x`, :attr:`y` and :attr:`z`.
		axis : np.ndarray | list[int | float] | str, optional
			This attribute specifies the axis of rotation for hinge Joints and the direction of 
			translation for Slides.
		range : np.ndarray | list[int | float], optional
			The reference position or angle of the Joint. It defines the joint value corresponding 
			to the initial model configuration.
		ref : int | float, optional
			The reference position or angle of the Joint. It defines the joint value corresponding 
			to the initial model configuration.
		frictionloss : int | float, optional
			Friction loss due to dry friction. This value is the same for all degrees of freedom 
			created by this Joint.
		name : str | None, optional
			The user specified name. In the case of a naming conflict the name will be altered by 
			an enumeration scheme.
		**kwargs
			Keyword arguments are passed to ``super().__init__``.
		"""
		axis = blue.utils.geometry.Vector.get_axis(axis)
		self.axis         = axis
		self.range        = range
		self.ref          = ref
		self.frictionloss = frictionloss
		super().__init__(pos=pos, 
				 name=name, 
				 x=x, 
				 y=y, 
				 z=z, 
				 **kwargs)


	@property
	def axis(self) -> np.ndarray:
		"""
		This attribute specifies the axis of rotation for hinge Joints and the direction of translation for Slides.
		
		Returns
		-------
		np.ndarray
		"""
		return self._axis


	@axis.setter
	@blue.restrict
	def axis(self, axis: np.ndarray|list[int|float]) -> None:
		"""
		This attribute specifies the axis of rotation for hinge Joints and the direction of translation for Slides.
		
		Parameters
		----------
		axis : np.ndarray | list[int | float]
			The axis which is assigned
		"""
		self._axis = np.array(axis, np.float32)


	@property
	def range(self) -> np.ndarray:
		"""
		The reference position or angle of the Joint. It defines the joint value corresponding to the initial model configuration.
		
		Returns
		-------
		np.ndarray
		"""
		return self._range


	@range.setter
	@blue.restrict
	def range(self, range: np.ndarray|list[int|float]) -> None:
		"""
		The reference position or angle of the Joint. It defines the joint value corresponding to the initial model configuration.
		
		Parameters
		----------
		range : np.ndarray | list[int | float]
			The range which is assigned
		"""
		self._range = np.array(range, np.float32)


	@property
	def ref(self) -> float:
		"""
		The reference position or angle of the Joint. It defines the joint value corresponding to the initial model configuration.
		
		Returns
		-------
		float
		"""
		return self._ref


	@ref.setter
	@blue.restrict
	def ref(self, ref: int|float) -> None:
		"""
		The reference position or angle of the Joint. It defines the joint value corresponding to the initial model configuration.
		
		Parameters
		----------
		ref : int | float
			The ref which is assigned
		"""
		self._ref = float(ref)


	@property
	def frictionloss(self) -> float:
		"""
		Friction loss due to dry friction. This value is the same for all degrees of freedom created by this Joint.
		
		Returns
		-------
		float
		"""
		return self._frictionloss


	@frictionloss.setter
	@blue.restrict
	def frictionloss(self, frictionloss: int|float) -> None:
		"""
		Friction loss due to dry friction. This value is the same for all degrees of freedom created by this Joint.
		
		Parameters
		----------
		frictionloss : int | float
			The xxx wfrictionlossh is assigned
		"""
		self._frictionloss = float(frictionloss)



class Slide(blue.SlideType, BaseJoint):

	"""
	Most attribute descriptions are partially taken from `Mujoco <https://mujoco.readthedocs.io/en/latest/XMLreference.html#body-joint>`__.
	"""
	
	@blue.restrict
	def __init__(self, 
		     pos:          np.ndarray|list[int|float]     = [0., 0., 0.], 
		     axis:         np.ndarray|list[int|float]|str = [0., 0., 1.], 
		     range:        np.ndarray|list[int|float]     = [0., 0.], 
		     ref:          int|float = 0, 
		     frictionloss: int|float = 0., 
		     name:         str|None  = None, 
		     x:            int|float|np.int32|np.int64|np.float32|np.float64|None = None, 
		     y:            int|float|np.int32|np.int64|np.float32|np.float64|None = None, 
		     z:            int|float|np.int32|np.int64|np.float32|np.float64|None = None, 
		     **kwargs):
		"""
		Parameters
		----------
		pos : np.ndarray | list[int | float], optional
			Represents the position of the object. Changing this attribute also changes the properties 
			:attr:`x`, :attr:`y` and :attr:`z`.
		axis : np.ndarray | list[int | float] | str, optional
			This attribute specifies the axis of rotation for hinge Joints and the direction of translation for Slides.
		range : np.ndarray | list[int | float], optional
			The reference position or angle of the Joint. It defines the joint value corresponding to the initial model configuration.
		ref : int | float, optional
			The reference position or angle of the Joint. It defines the joint value corresponding to the initial model configuration.
		frictionloss : int | float, optional
			Friction loss due to dry friction. This value is the same for all degrees of freedom created by this Joint.
		name : str | None, optional
			The user specified name. In the case of a naming conflict the name will be altered by an 
			enumeration scheme.
		**kwargs
			Keyword arguments are passed to ``super().__init__``.
		"""
		axis = blue.utils.geometry.Vector.get_axis(axis)
		self.axis         = axis
		self.range        = range
		self.ref          = ref
		self.frictionloss = frictionloss
		super().__init__(pos=pos, 
				 name=name, 
				 x=x, 
				 y=y, 
				 z=z, 
				 **kwargs)


	@property
	def axis(self) -> np.ndarray:
		"""
		This attribute specifies the axis of rotation for hinge Joints and the direction of translation for Slides.
		
		Returns
		-------
		np.ndarray
		"""
		return self._axis


	@axis.setter
	@blue.restrict
	def axis(self, axis: np.ndarray|list[int|float]) -> None:
		"""
		This attribute specifies the axis of rotation for hinge Joints and the direction of translation for Slides.
		
		Parameters
		----------
		axis : np.ndarray | list[int | float]
			The axis which is assigned
		"""
		self._axis = np.array(axis, np.float32)


	@property
	def range(self) -> np.ndarray:
		"""
		The reference position or angle of the Joint. It defines the joint value corresponding to the initial model configuration.
		
		Returns
		-------
		np.ndarray
		"""
		return self._range


	@range.setter
	@blue.restrict
	def range(self, range: np.ndarray|list[int|float]) -> None:
		"""
		The reference position or angle of the Joint. It defines the joint value corresponding to the initial model configuration.
		
		Parameters
		----------
		range : np.ndarray | list[int | float]
			The range which is assigned
		"""
		self._range = np.array(range, np.float32)


	@property
	def ref(self) -> float:
		"""
		The reference position or angle of the Joint. It defines the joint value corresponding to the initial model configuration.
		
		Returns
		-------
		float
		"""
		return self._ref


	@ref.setter
	@blue.restrict
	def ref(self, ref: int|float) -> None:
		"""
		The reference position or angle of the Joint. It defines the joint value corresponding to the initial model configuration.
		
		Parameters
		----------
		ref : int | float
			The ref which is assigned
		"""
		self._ref = float(ref)


	@property
	def frictionloss(self) -> float:
		"""
		Friction loss due to dry friction. This value is the same for all degrees of freedom created by this Joint.
		
		Returns
		-------
		float
		"""
		return self._frictionloss


	@frictionloss.setter
	@blue.restrict
	def frictionloss(self, frictionloss: int|float) -> None:
		"""
		Friction loss due to dry friction. This value is the same for all degrees of freedom created by this Joint.
		
		Parameters
		----------
		frictionloss : int | float
			The xxx wfrictionlossh is assigned
		"""
		self._frictionloss = float(frictionloss)



class Ball(blue.BallType, BaseJoint):

	"""
	Most attribute descriptions are partially taken from `Mujoco <https://mujoco.readthedocs.io/en/latest/XMLreference.html#body-joint>`__.
	"""
	
	@blue.restrict
	def __init__(self, 
		     pos:          np.ndarray|list[int|float] = [0., 0., 0.], 
		     range:        np.ndarray|list[int|float] = [0., 0.], 
		     frictionloss: int|float                  = 0., 
		     name:         str|None                   = None, 
		     x:            int|float|np.int32|np.int64|np.float32|np.float64|None = None, 
		     y:            int|float|np.int32|np.int64|np.float32|np.float64|None = None, 
		     z:            int|float|np.int32|np.int64|np.float32|np.float64|None = None, 
		     **kwargs):
		"""
		Parameters
		----------
		pos : np.ndarray | list[int | float], optional
			Represents the position of the object. Changing this attribute also changes the properties 
			:attr:`x`, :attr:`y` and :attr:`z`.
		range : np.ndarray | list[int | float], optional
			The reference position or angle of the Joint. It defines the joint value corresponding to the initial model configuration.
		frictionloss : int | float, optional
			Friction loss due to dry friction. This value is the same for all degrees of freedom created by this Joint.
		name : str | None, optional
			The user specified name. In the case of a naming conflict the name will be altered by an 
			enumeration scheme.
		x : int | float |np.int32 | np.int64 | np.float32 | np.float64 | None, optional
			If `pos` is not specified, this argument sets the X position coordinate.
		y : int | float |np.int32 | np.int64 | np.float32 | np.float64 | None, optional
			If `pos` is not specified, this argument sets the Y position coordinate.
		z : int | float |np.int32 | np.int64 | np.float32 | np.float64 | None, optional
			If `pos` is not specified, this argument sets the Z position coordinate.
		**kwargs
			Keyword arguments are passed to ``super().__init__``.
		"""
		self.range        = range
		self.frictionloss = frictionloss
		super().__init__(pos=pos, 
				 name=name, 
				 x=x, 
				 y=y, 
				 z=z, 
				 **kwargs)


	@property
	def range(self) -> np.ndarray:
		"""
		The reference position or angle of the Joint. It defines the joint value corresponding to the initial model configuration.
		
		Returns
		-------
		np.ndarray
		"""
		return self._range


	@range.setter
	@blue.restrict
	def range(self, range: np.ndarray|list[int|float]) -> None:
		"""
		The reference position or angle of the Joint. It defines the joint value corresponding to the initial model configuration.
		
		Parameters
		----------
		range : np.ndarray | list[int | float]
			The range which is assigned
		"""
		self._range = np.array(range, np.float32)


	@property
	def frictionloss(self) -> float:
		"""
		Friction loss due to dry friction. This value is the same for all degrees of freedom created by this Joint.
		
		Returns
		-------
		float
		"""
		return self._frictionloss


	@frictionloss.setter
	@blue.restrict
	def frictionloss(self, frictionloss: int|float) -> None:
		"""
		Friction loss due to dry friction. This value is the same for all degrees of freedom created by this Joint.
		
		Parameters
		----------
		frictionloss : int | float
			The xxx which is assigned
		"""
		self._frictionloss = float(frictionloss)



class Free(blue.FreeType, BaseJoint):

	"""
	Most attribute descriptions are partially taken from `Mujoco <https://mujoco.readthedocs.io/en/latest/XMLreference.html#body-joint>`__.
	"""
	
	def __init__(self, 
		     **kwargs):
		"""
		Parameters
		----------
		**kwargs
			Keyword arguments are passed to ``super().__init__``.
		"""
		super().__init__(**kwargs)



class Joint(Hinge):

	"""
	Dummy for :class:`Hinge`
	"""

	@property
	def type(self) -> None:
		"""
		Dummy for :class:`Hinge`
		
		Returns
		-------
		str
			``'hinge'``
		"""
		return None



JOINT_THINGS = {None:   Joint, 
	       'hinge': Hinge, 
	       'slide': Slide,
	       'ball':  Ball, 
	       'free':  Free}
