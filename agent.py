"""
To make blueprints amneable for multi agent settings the :class:`Agent` class provides easily structured
access to the relevant Things for observation (:class:`Sensors <blueprints.sensors.BaseSensor>` and :class:`Cameras <blueprints.Camera>`) 
and action (:class:`Actuators <blueprints.actuators.General>`).

Agents are used similar to :class:`Bodies <blueprints.bodies.Body>` with some minor differences.
The child attributes for normal Things map only to direct children, but for an :class:`Agent` instance 
the attributes :attr:`actuators <Agent.actuators>`, :attr:`sensors <Agent.sensors>` and :attr:`cameras <Agent.cameras>` 
map to all descendants as a more intuitive shortcut.
All observations from Sensors and Cameras are bundeled in an :attr:`observation <Agent.observation>` attribute and 
all actions can be applied through the :attr:`force <Agent.force>` and :attr:`activation <Agent.activation>` attribute.

.. code-block:: mxml
	:caption: Agent XML structure
	
	<sensors>
		<force site='site_1'>
		<force site='site_2'>
	</sensors>
	<actuators>
		<general body='AGENT:body_1' dynprm='filter'>
		<general body='body_2'>
	</actuators>
	<body name='AGENT:body_1'>
		<body name='body_2'>
			<site name='site_1'>
		</body>
		<site name='site_2'>
		<camera name='cam'>
		<joint name='joint'>
	</body>

.. code-block::
	:caption: Agent Interface

	>>> agent.observation
	{'<Camera>cam': np.ndarray([...], shape=(width, height)), 
	 '<Force>_(0)': np.ndarray([0.]), 
	 '<Force>_(1)': np.ndarray([0.])}
	>>> agent.action_shape
	{'force': 1, 
	 'activation': 1}
	>>> agent.force = np.darray([...])
	>>> agent.activation = = np.darray([...])



Additionally shapes for observations and actions are available as attributes as well.
"""

import blueprints as blue
import numpy as np



class Agent(blue.AgentType, blue.Body):
	"""
	This class emulates a Body which serves as an interface for 
	sensors, cameras and actuators of an Agent. It provides uniform 
	access to the observation and action relevant Things.
	"""
	def __str__(self) -> str:
		"""
		Returns
		-------
		str
			A representation of the Agent. The xml will be parsed as :class:`Body <blue.bodies.Body>`.
		"""
		return f'<agent/body name="{self.name}">'


	@property
	def name(self) -> str:
		"""
		The returned name might differ from the user specification by an enumerations 
		scheme that is applied if two Things of the same type in a kinematic tree share 
		the same name.
		
		Returns
		-------
		str
			possibly extended name of the Thing
		"""
		if self._name_scope is not None:
			return f"AGENT:{self._name_scope.name(self).replace('AGENT:', '')}"
		else:
			return f"AGENT:{self._name.replace('AGENT:', '')}"

	# AGENT PROPERTIES

	@property
	def cameras(self) -> blue.ViewType:
		"""
		Instead of returning just the children all descending Cameras are returned.
		
		Returns
		-------
		View
			The view is routed to all cameras of the agent (instead of just those that follow directly in the kinematic hierarchy).
		"""
		cameras = self.descendants['cameras']['descendants']
		return blue.View(cameras, 'cameras', self)


	@property
	def sensors(self) -> blue.ViewType:
		"""
		Instead of returning just the children all descending Sensors are returned.
		
		Returns
		-------
		View
			The view is routed to all sensors of the agent (instead of just those that follow directly in the kinematic hierarchy).
		"""
		sensors = self.descendants['sensors']['descendants']
		return blue.View(sensors, 'sensors', self)


	@property
	def actuators(self) -> blue.ViewType:
		"""
		Instead of returning just the children all descending Actuators are returned.
		
		Returns
		-------
		View
			The view is routed to all actuators of the agent (instead of just those that follow directly in the kinematic hierarchy).
		"""
		actuators = self.descendants['actuators']['descendants']
		return blue.View(actuators, 'actuators', self)


	@property
	def sensor_observation_shape(self) -> dict:
		"""
		The dictionary contains the names of all sensors and their shapes.
		
		Returns
		-------
		dict
		"""
		if not self.root._built:
			raise Exception('Agent.observation_shape is only accessable after the World has been built.')
		return {f'<{sensor.__class__.__name__}>{sensor.name}': (sensor.DIMENSIONS,) \
			for sensor in self.descendants['sensors']['descendants']}


	@property
	def camera_observation_shape(self) -> dict:
		"""
		The dictionary contains the names of all cameras and their shapes.
		
		Returns
		-------
		dict
		"""
		if not self.root._built:
			raise Exception('Agent.observation_shape is only accessable after the World has been built.')
		return {f'<{camera.__class__.__name__}>{camera.name}': tuple(camera.resolution) \
			for camera in self.descendants['cameras']['descendants']}


	@property
	def observation_shape(self) -> dict:
		"""
		The dictionary contains the names of all observables and their shapes.
		
		Returns
		-------
		dict
		"""
		if not self.root._built:
			raise Exception('Agent.observation_shape is only accessable after the World has been built.')
		cameras = self.camera_observation_shape
		sensors = self.sensor_observation_shape
		return {**cameras, **sensors}


	@property
	def sensor_observation(self) -> dict:
		"""
		The dictionary contains the name of all sensor observables and their data.
		
		Returns
		-------
		dict
		"""
		if not self.root._built:
			raise Exception('Agent.observation is only accessable after the World has been built.')
		return {f'<{sensor.__class__.__name__}>{sensor.name}': sensor.observation \
			for sensor in self.descendants['sensors']['descendants']}


	@property
	def camera_observation(self) -> dict:
		"""
		The dictionary contains the name of all camera observables and their data.
		
		Returns
		-------
		dict
		"""
		if not self.root._built:
			raise Exception('Agent.observation is only accessable after the World has been built.')
		return {f'<{camera.__class__.__name__}>{camera.name}': camera.observation \
			for camera in self.descendants['cameras']['descendants']}
		
	@property
	def observation(self) -> dict:
		"""
		The dictionary contains the name of all observables and their data.
		
		Returns
		-------
		dict
		"""
		if not self.root._built:
			raise Exception('Agent.observation is only accessable after the World has been built.')
		cameras = self.camera_observation
		sensors = self.sensor_observation
		return {**cameras, **sensors}


	@property
	def activation(self) -> list:
		"""
		All activations of Actuators that have an activation state. 
		Actuators with ``'dyntype' == 'none'`` do not have an activation.
		
		Returns
		-------
		list
		"""
		activation = self.actuators.activation
		return [x for x in activation if x is not None]


	@activation.setter
	@blue.restrict
	def activation(self, 
		       activation: list[int|float]|np.ndarray) -> None:
		"""
		A 1-D vector of activations for the actuators activation states.
		
		Parameters
		----------
		activation : list[int | float] | np.ndarray
		"""
		actuators = self.actuators
		mask = [x is not None for x in activation]
		for i, (x, m) in enumerate(zip(activation, mask)):
			if m:
				actuators[i] = x


	@property
	def force(self) -> list:
		"""
		All forces of Actuators of the Agent
		
		Returns
		-------
		list
		"""
		return self.actuators.force


	@force.setter
	@blue.restrict
	def force(self, 
		  force: list[int|float]|np.ndarray) -> None:
		"""
		Parameters
		----------
		force : list[int | float] | np.ndarray
			A 1-D vector of forces to be applied to the actuators
		"""
		actuators = self.actuators
		for i, f in enumerate(force):
			actuators[i].force = float(f)


	@property
	def action_shape(self) -> dict:
		"""
		The shapes of both action types (activations and forces)
		
		Returns
		-------
		dict
		"""
		if not self.root._built:
			raise Exception('Agent.action_shape is only accessable after the World has been built.')
		activation = self.activation
		force      = self.force
		return {'activation': len(activation), 'force': len(force)}
