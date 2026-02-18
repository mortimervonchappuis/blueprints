import blueprints as blue
import xml.etree.ElementTree as xml
import numpy as np
import mujoco



class BaseSensor(blue.SensorType, blue.thing.BaseThing):

	"""
	Sensors can be used to as an interface for the agent to the World. Sensors that are part of the 
	kinematic tree of an Agent are included in the Agents observations.
	
	Most attribute descriptions are partially taken from `Mujoco <https://mujoco.readthedocs.io/en/latest/XMLreference.html#sensor>`__.
	"""

	@blue.restrict
	def __init__(self, 
		     noise:     float|int           = 0., 
		     cutoff:    float|int           = 0., 
		     name:      str|None            = None, 
		     reference: blue.ThingType|None = None, 
		     **kwargs):
		"""
		Parameters
		----------
		noise : float | int, optional
		   	The standard deviation of zero-mean Gaussian noise added to the sensor output.
		cutoff : float | int, optional
			When this value is positive, it limits the absolute value of the sensor output.
		name : str | None, optional
			The user specified name might potentially be altered to avoid a naming conflict 
			by appending an enumeration scheme.
		reference : blue.ThingType | None, optional
			The reference of the Sensor attribute is the parent of the Sensor which is translated 
			to the :attr:`site <SiteSensor.site>`, :attr:`joint <JointSensor.joint>` or 
			:attr:`actuator <ActuatorSensor.actuator>` attributes for xml.
		**kwargs
			Keyword arguments are passed to ``super().__init__``.
		
		Raises
		------
		TypeError
			If the reference type is not valid an error is raised.
		"""
		if not isinstance(reference, tuple(self._REFERENCE_TYPES)) and reference is not None:
			raise TypeError(f'The Sensor reference must be of the types ({", ".join(map(str, self._REFERENCE_TYPES))}), got ({type(reference)}) instead.')
		self.noise  = noise
		self.cutoff = cutoff
		super().__init__(name=name, **kwargs)
		self.reference = reference


	@property
	def observation(self) -> np.ndarray:
		"""
		The live sensor data from the simulation.
		
		Returns
		-------
		np.ndarray
		"""
		if not hasattr(self, '_index'):
			raise Exception('Sensor must first be build by a World before observations are available.')
		else:
			return self.root._mj_data.sensordata[self._index:self._index + self.DIMENSIONS].copy()


	@blue.restrict
	def _build(self, 
		   parent, 
		   world:    blue.WorldType, 
		   indicies: dict, 
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
			The builded xml element of the Sensor.
		"""
		self._index          = indicies['sensors']
		indicies['sensors'] += self.DIMENSIONS
		self._xml_root       = xml.SubElement(world._xml_sensor, 
						      self.type,  
						      **self._mujoco_specs(kwargs))
		return self._xml_root


	@blue.restrict
	@classmethod
	def _from_xml_element(cls, xml_element: xml.Element) -> blue.ThingType:
		"""
		This method reconstructs a Thing from an xml element. If any argument for 
		an inheriting class has to be set manually this method must be overwritten.
		
		Parameters
		----------
		xml_element : xml.Element
			The xml element from which a Thing is reconstructed.
		
		Returns
		-------
		blue.ThingType
			The reconstructed Sensor.
		"""
		init_args, post_args, rest_args = cls._xml_element_args(xml_element)
		sensor_type = xml_element.tag
		sensor = object.__new__(blue.REGISTER.SENSOR_THINGS[sensor_type])
		sensor.__init__(**init_args)
		for key, val in post_args.items():
			setattr(sensor, key, val)
		return sensor


	@property
	def reference(self):
		"""
		The reference of the Sensor attribute is the parent of the Sensor which is translated 
		to the :attr:`site <SiteSensor.site>`, :attr:`joint <JointSensor.joint>` or 
		:attr:`actuator <ActuatorSensor.actuator>` attributes for xml.

		Returns
		-------
		blue.ThingType
			The sensors refferenced Thing
		"""
		return self.parent


	@reference.setter
	@blue.restrict
	def reference(self, reference: blue.NodeThingType|None):
		"""
		Parameters
		----------
		reference : blue.NodeThingType | None
			The reference of the Sensor attribute is the parent of the Sensor which is translated 
			to the :attr:`site <SiteSensor.site>`, :attr:`joint <JointSensor.joint>` or 
			:attr:`actuator <ActuatorSensor.actuator>` attributes for xml.
		"""
		if self.parent is not None:
			self.parent.detach(self)
		if reference is not None:
			reference.attach(self, copy=False)

	# MUJOCO PROPERTIES

	@property
	def type(self):
		"""
		This is a derived attribute, which is used to specify the ``type`` attribute in the 
		mujoco ``sensor`` tag.

		Returns
		-------
		str
			The type is the lower case name of the Sensor class.
		"""
		return self.__class__.__name__.lower()


	@property
	def noise(self) -> float:
		"""
		The standard deviation of zero-mean Gaussian noise added to the sensor output.
		
		Returns
		-------
		float
		"""
		return self._noise


	@noise.setter
	@blue.restrict
	def noise(self, noise: float|int) -> None:
		"""
		Parameters
		----------
		noise : float | int
			The standard deviation of zero-mean Gaussian noise added to the sensor output.
		"""
		self._noise = float(noise)


	@property
	def cutoff(self) -> float:
		"""
		When this value is positive, it limits the absolute value of the sensor output.
		
		Returns
		-------
		float
		"""
		return self._cutoff


	@cutoff.setter
	@blue.restrict
	def cutoff(self, cutoff: float|int) -> None:
		"""
		Parameters
		----------
		cutoff : float | int
			When this value is positive, it limits the absolute value of the sensor output.
		"""
		self._cutoff = float(cutoff)



# INTERMEDIATE SENSORS



class SiteSensor(blue.SiteSensorType, BaseSensor):

	"""
	Intermediate Sensor class used by Sensors that reference :class:`Sites <blueprints.sites.BaseSite>`.
	"""
	
	@property
	def site(self):
		"""
		The parent name of the Sensor which is translated to this attribute for xml construction.
		
		Returns
		-------
		blue.ThingType
		"""
		if self.reference:
			return self.reference.name
		else:
			return None



class JointSensor(blue.JointSensorType, BaseSensor):

	"""
	Intermediate Sensor class used by Sensors that reference :class:`Joints <blueprints.joints.BaseJoint>`.
	"""
	
	@property
	def joint(self):
		"""
		The parent name of the Sensor which is translated to this attribute for xml construction.
		
		Returns
		-------
		blue.ThingType
		"""
		if self.reference:
			return self.reference.name
		else:
			return None



class ActuatorSensor(blue.ActuatorSensorType, BaseSensor):

	"""
	Intermediate Sensor class used by Sensors that reference :class:`Actuators <blueprints.actuators.BaseActuator>`.
	"""
	
	@property
	def actuator(self):
		"""
		The parent name of the Sensor which is translated to this attribute for xml construction.
		
		Returns
		-------
		blue.ThingType
		"""
		if self.reference:
			return self.reference.name
		else:
			return None



# INFO LASER



class InfoLaser(blue.InfoLaserType, SiteSensor):

	"""
	The InfoLaser is not a mujoco native object but a derived blueprints object, 
	that will not appear in xml. Instead it uses the mujoco.mj_ray subroutine to 
	cast a ray from the Sensors Site and returns the intersected object and the 
	distance to it.
	"""
	
	@blue.restrict
	def __init__(self, 
		     axis:      np.ndarray|list[int|float] = [0., 0., 1.], 
		     noise:     float|int                  = 0., 
		     cutoff:    float|int                  = 0., 
		     name:      str|None                   = None, 
		     reference: blue.ThingType|None        = None, 
		     **kwargs):
		"""
		Parameters
		----------
		axis : np.ndarray
			The local axis along which the InfoLasers ray will be casted.
		noise : float | int, optional
		   	The standard deviation of zero-mean Gaussian noise added to the sensor output.
		cutoff : float | int, optional
			When this value is positive, it limits the absolute value of the sensor output.
		name : str | None, optional
			The user specified name might potentially be altered to avoid a naming conflict 
			by appending an enumeration scheme.
		reference : blue.ThingType | None, optional
			The reference of the Sensor attribute is the parent of the Sensor which is translated 
			to the :attr:`site <SiteSensor.site>`, :attr:`joint <JointSensor.joint>` or 
			:attr:`actuator <ActuatorSensor.actuator>` attributes for xml.
		**kwargs
			Keyword arguments are passed to ``super().__init__``.
		
		Raises
		------
		TypeError
			If the reference type is not valid an error is raised.
		"""
		self.axis = axis
		super().__init__(noise=noise, 
				 cutoff=cutoff, 
				 name=name, 
				 reference=reference, 
				 **kwargs)


	@property
	def observation(self) -> dict:
		"""
		Computes the observation on runtime.
		"""
		if not self.root._launched:
			raise Exception('Sensor must first be build by a World before observations are available.')
		else:
			R  = self.parent.global_rotation_matrix
			ID = np.array([-1], np.int32)
			distance = mujoco.mj_ray(m=self.root._mj_model, 
						 d=self.root._mj_data, 
						 pnt=self.parent.global_pos, 
						 vec=R @ self.axis, 
						 geomgroup=None, 
						 flg_static=1, 
						 bodyexclude=-1, 
						 geomid=ID)
			if distance == -1:
				geom = None
			else:
				geom = self.root.descendants['geoms']['descendants'][ID[0]]
			return {'distance': distance, 'geom': geom}



	@property
	def axis(self) -> np.ndarray:
		"""
		The local axis along which the InfoLasers ray will be casted.
		
		Returns
		-------
		np.ndarray
		"""
		return self._axis


	@axis.setter
	@blue.restrict
	def axis(self, 
		 axis: np.ndarray|list[int|float]) -> None:
		"""
		Parameters
		-------
		axis : np.ndarray
			The local axis along which the InfoLasers ray will be casted.
		"""
		self._axis = np.array(axis, dtype=np.float32)


	@blue.restrict
	def _build(self, 
		   parent, 
		   world, 
		   indicies, 
		   **kwargs):
		"""
		InfoLasers are not part of mujoco xml.
		
		Parameters
		----------
		parent : xml.etree.ElementTree.Element
			The parent of the Thing
		world : blue.WorldType
			The World from which the initial _meth:`blueprints.world.World.build` method was called.
		"""
		pass


# SITE SENSORS



class Touch(SiteSensor):
	DIMENSIONS = 1



class Accelerometer(SiteSensor):
	DIMENSIONS = 3



class Velocimeter(SiteSensor):
	DIMENSIONS = 3



class Gyro(SiteSensor):
	DIMENSIONS = 3



class Force(SiteSensor):
	DIMENSIONS = 3



class Torque(SiteSensor):
	DIMENSIONS = 3



class Rangefinder(SiteSensor):
	DIMENSIONS = 1



# JOINT SENSORS



class JointPos(JointSensor):
	DIMENSIONS = 1



class JointVel(JointSensor):
	DIMENSIONS = 1



class JointLimitPos(JointSensor):
	DIMENSIONS = 1



class JointLimitVel(JointSensor):
	DIMENSIONS = 1



class JointLimitFrc(JointSensor):
	DIMENSIONS = 1



class BallQuat(JointSensor):
	DIMENSIONS = 4



class BallAngVel(JointSensor):
	DIMENSIONS = 3



# ACTUATOR SENSOR



class ActuatorPos(ActuatorSensor):
	DIMENSIONS = 1



class ActuatorVel(ActuatorSensor):
	DIMENSIONS = 1



class ActuatorFrc(ActuatorSensor):
	DIMENSIONS = 1



SENSOR_THINGS  = {'touch':         Touch, 
		  'accelerometer': Accelerometer, 
		  'velocimeter':   Velocimeter, 
		  'gyro':          Gyro, 
		  'force':         Force, 
		  'torque':        Torque, 
		  'rangefinder':   Rangefinder, 
		  'jointpos':      JointPos, 
		  'jointvel':      JointVel, 
		  'jointlimitpos': JointLimitPos, 
		  'jointlimitvel': JointLimitVel, 
		  'jointlimitfrc': JointLimitFrc, 
		  'ballquat':      BallQuat, 
		  'ballangvel':    BallAngVel, 
		  'actuatorpos':   ActuatorPos, 
		  'actuatorvel':   ActuatorVel, 
		  'actuatorfrc':   ActuatorFrc}
