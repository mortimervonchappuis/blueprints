"""
A :class:`Body` is the primary Thing used to build the structure of a kinematic tree. This is done by 
attaching them to other bodies as well as other bodies to them. Bodies do have a position and an orientation 
but no further physical properties like a shape, a size or mass. Positions and orientations are local meaning 
that a body is placed relative to its parents frame of reference.To flesh out a :class:`Body` the following 
Things can be attached to it:

* :class:`Geom <blueprints.geoms.BaseGeom>`: This Thing is used to fill in the Body with matter. Geoms have a shape, mass, friction etc.

* :class:`Site <blueprints.sites.BaseSite>`: Sites similar to Geoms have shape, but do not effect the simulation physics. Instead :class:`Sensor <blueprints.sensors.BaseSensor>` instances as well as :class:`Actuator <blueprints.actuators.BaseActuator>` can be attached to it.

* :class:`Camera <blueprints.camera.Camera>`: If attached to the Body, the Camera can be view trough :meth:`World.view <blueprints.world.World.view>` by setting the :attr:`rendering <blueprints.camera.Camera.rendering>` camera value to the Cameras name. If the Body is included in an :class:`Agent <blueprints.agent.Agent>` (not existing right now) the camera will be available in the Agents observations.

* :class:`Placeholder <blueprints.placeholder.Placeholder>`: A placeholder can be used to specify a position and orientation relative to the Body to which it is attached, that is of special interest. If a Thing is attached to a placeholder it is instead attached to its parent in a position and orientation, that matches attachment to the placeholder.

* :class:`Actuator <blueprints.actuators.BaseActuator>`: Actuators are used to apply force to the kinematic tree at the node they are attached to. If the Body is included in an :class:`Agent <blueprints.agent.Agent>` (not yet implemented) the actuators input will appear in the Agents action space.

* :class:`Light <blueprints.light.Light>`: Light sources illuminate the space around them, but are not necessary for the model to be visible, since a base luminescence is presence always present.

* :class:`Joint <blueprints.joints.BaseJoint>`: This Thing is used to move the :class:`Body`:. If a force is applied to the Body it moves along the degrees of freedom defined by the Joint.

.. note::
	Unintuitively Bodies are not connected with each other via a Joint. Instead a Joint attached to a 
	body defines the way in which a Body can be moved w.r.t. to the Bodies parent.

A kinematic tree can either be constructed bottom up or top down. The following two examples are used to 
create a kinematic tree of this structure:

.. code-block:: mxml
	:caption: XML Structure

	<body name="tree">
	    <body pos="0.0 0.0 8.0" name="stem">
	        <body euler="0.0 -0.7853982 0.0" pos="0.0 0.0 4.0" name="branch_(0)">
	            <body euler="0.0 -0.7853982 0.0" pos="0.0 0.0 2.0" name="twig_(0)">
	                <body euler="0.0 -0.7853982 0.0" pos="0.0 0.0 1.0" name="leaf_(0)" />
	                <body euler="0.0 0.7853982 0.0" pos="0.0 0.0 1.0" name="leaf_(1)" />
	            </body>
	            <body euler="0.0 0.7853982 0.0" pos="0.0 0.0 2.0" name="twig_(1)">
	                <body euler="0.0 -0.7853982 0.0" pos="0.0 0.0 1.0" name="leaf_(2)" />
	                <body euler="0.0 0.7853982 0.0" pos="0.0 0.0 1.0" name="leaf_(3)" />
	            </body>
	        </body>
	        <body euler="0.0 0.7853982 0.0" pos="0.0 0.0 4.0" name="branch_(1)">
	            <body euler="0.0 -0.7853982 0.0" pos="0.0 0.0 2.0" name="twig_(2)">
	                <body euler="0.0 -0.7853982 0.0" pos="0.0 0.0 1.0" name="leaf_(4)" />
	                <body euler="0.0 0.7853982 0.0" pos="0.0 0.0 1.0" name="leaf_(5)" />
	            </body>
	            <body euler="0.0 0.7853982 0.0" pos="0.0 0.0 2.0" name="twig_(3)">
	                <body euler="0.0 -0.7853982 0.0" pos="0.0 0.0 1.0" name="leaf_(6)" />
	                <body euler="0.0 0.7853982 0.0" pos="0.0 0.0 1.0" name="leaf_(7)" />
	            </body>
	        </body>
	    </body>
	</body>

.. code-block:: py
	:caption: Bottom-Up

	>>> tree = blue.Body(name='tree')
	>>> stem = blue.Body(name='stem',   pos=[0, 0, 8])
	>>> bran = blue.Body(name='branch', pos=[0, 0, 4])
	>>> twig = blue.Body(name='twig',   pos=[0, 0, 2])
	>>> leaf = blue.Body(name='leaf',   pos=[0, 0, 1])
	>>> angle = TAU/8
	>>> # starting at the leafs
	>>> twig.attach(leaf.rotate(beta=-angle), leaf.rotate(beta=angle))
	>>> bran.attach(twig.rotate(beta=-angle), twig.rotate(beta=angle))
	>>> stem.attach(bran.rotate(beta=-angle), bran.rotate(beta=angle))
	>>> tree.attach(stem)

.. code-block:: py
	:caption: Top-Down

	>>> tree = blue.Body(name='tree')
	>>> stem = blue.Body(name='stem',   pos=[0, 0, 0])
	>>> bran = blue.Body(name='branch', pos=[0, 0, 4])
	>>> twig = blue.Body(name='twig',   pos=[0, 0, 2])
	>>> leaf = blue.Body(name='leaf',   pos=[0, 0, 1])
	>>> angle = TAU/8
	>>> # starting at the stem
	>>> tree.attach(stem)
	>>> tree.bodies.attach(bran.rotate(beta=-angle), bran.rotate(beta=angle))
	>>> tree.bodies.bodies.attach(twig.rotate(beta=-angle), twig.rotate(beta=angle))
	>>> tree.bodies.bodies.bodies.attach(leaf.rotate(beta=-angle), leaf.rotate(beta=angle))
"""

import xml.etree.ElementTree as xml
import numpy  as np
import blueprints as blue



class Body(blue.BodyType, blue.thing.MoveableThing, blue.thing.NodeThing):
	"""
	This class is available through the shortcut :class:`blueprints.Body <Body>`.

	Attributes from base classes :class:`MoveableThing <blueprints.thing.moveable.MoveableThing>` and :class:`NodeThing <blueprints.thing.node.NodeThing>` are inherited.
	
	Attributes
	----------
	bodies, cameras, geoms, sites, joints, lights, cameras, actuators, placeholders : :class:`View <blueprints.utils.view.View>`
		All children that have been attached to the Body are retrieved by the attribute as a :class:`View <blueprints.utils.view.View>`.
	"""
	@blue.restrict
	def __init__(self, 
		     pos:          list[int|float]|np.ndarray = [0., 0., 0.], 
		     alpha:        int|float|None = 0., 
		     beta:         int|float|None = 0., 
		     gamma:        int|float|None = 0., 
		     geoms:        list[blue.ThingType]|blue.ThingType|None = None, 
		     sites:        list[blue.ThingType]|blue.ThingType|None = None, 
		     joints:       list[blue.ThingType]|blue.ThingType|None = None, 
		     bodies:       list[blue.ThingType]|blue.ThingType|None = None, 
		     lights:       list[blue.ThingType]|blue.ThingType|None = None, 
		     cameras:      list[blue.ThingType]|blue.ThingType|None = None, 
		     actuators:    list[blue.ThingType]|blue.ThingType|None = None, 
		     placeholders: list[blue.ThingType]|blue.ThingType|None = None, 
		     name:         str|None       = None, 
		     copy:         bool           = True, 
		     x:            int|float|np.int32|np.int64|np.float32|np.float64|None = None, 
		     y:            int|float|np.int32|np.int64|np.float32|np.float64|None = None, 
		     z:            int|float|np.int32|np.int64|np.float32|np.float64|None = None, 
		     quat:         list[int|float]|np.ndarray|None = None, 
		     **kwargs) -> None:
		"""
		Parameters
		----------
		pos : list[int | float] | np.ndarray, optional
			Represents the position of the object. Changing this attribute also changes the properties :attr:`x`, :attr:`y` and :attr:`z`.
		alpha : int | float | None, optional
			(Improper) euler angle of rotation around the x-axis in radian. Changing this value also changes the :attr:`euler` property.
		beta : int | float | None, optional
			(Improper) euler angle of rotation around the y-axis in radian. Changing this value also changes the :attr:`euler` property.
		gamma : int | float | None, optional
			(Improper) euler angle of rotation around the z-axis in radian. Changing this value also changes the :attr:`euler` property.
		geoms : list[blue.ThingType] | blue.ThingType | None, optional
			The ``geoms`` argument mus either be a list of Geoms or a single Geom.
		sites : list[blue.ThingType] | blue.ThingType | None, optional
			The ``sites`` argument mus either be a list of Sites or a single Site.
		joints : list[blue.ThingType] | blue.ThingType | None, optional
			The ``joints`` argument mus either be a list of Joints or a single Joint.
		bodies : list[blue.ThingType] | blue.ThingType | None, optional
			The ``bodies`` argument mus either be a list of Bodies or a single Body.
		lights : list[blue.ThingType] | blue.ThingType | None, optional
			The ``lights`` argument mus either be a list of Lights or a single Light.
		cameras : list[blue.ThingType] | blue.ThingType | None, optional
			The ``cameras`` argument mus either be a list of Cameras or a single Camera.
		actuators : list[blue.ThingType] | blue.ThingType | None, optional
			The ``actuators`` argument mus either be a list of Actuators or a single Actuator.
		placeholders : list[blue.ThingType] | blue.ThingType | None, optional
			The ``placeholders`` argument mus either be a list of Placeholders or a single Placeholder.
		name : str | None, optional
			The user specified name of the Body. In the case of a naming conflict the name is appended by an enumeration scheme.
		copy : bool, optional
			This argument indicates whether the children should be copied before they are attached. If copy if set to False this will result in graph cycles if a parent of the Body is attached resulting in further errors.
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
		pack = lambda x: x if isinstance(x, list) else [x]
		geoms        = pack(geoms)        if geoms        else []
		joints       = pack(joints)       if joints       else []
		bodies       = pack(bodies)       if bodies       else []
		lights       = pack(lights)       if lights       else []
		sites        = pack(sites)        if sites        else []
		cameras      = pack(cameras)      if cameras      else []
		actuators    = pack(actuators)    if actuators    else []
		placeholders = pack(placeholders) if placeholders else []
		self._geoms             = []
		self._joints            = []
		self._bodies            = []
		self._lights            = []
		self._sites             = []
		self._cameras           = []
		self._actuators         = []
		self._placeholders      = []
		self._targeting_lights  = []
		self._targeting_cameras = []
		self._CHILDREN  = {'lights':            {'type':     blue.LightType, 
							 'children': self._lights}, 
				   'joints':            {'type':     blue.JointType, 
							 'children': self._joints}, 
				   'geoms':             {'type':     blue.GeomType, 
							 'children': self._geoms}, 
				   'sites':             {'type':     blue.SiteType, 
							 'children': self._sites}, 
				   'cameras':           {'type':     blue.CameraType, 
							 'children': self._cameras}, 
				   'actuators':         {'type':     blue.ActuatorType, 
							 'children': self._actuators}, 
				   'placeholders':      {'type':     blue.PlaceholderType, 
							 'children': self._placeholders}, 
				   'bodies':            {'type':     blue.BodyType, 
							 'children': self._bodies}}
		self._CYCLE_REF = {'targeting_lights':  {'type':     blue.CameraType, 
							 'children': self._targeting_lights}, 
				   'targeting_cameras': {'type':     blue.CameraType, 
							 'children': self._targeting_cameras}}
		super().__init__(pos=pos, 
				 x=x, 
				 y=y, 
				 z=z, 
				 alpha=alpha, 
				 beta=beta, 
				 gamma=gamma, 
				 quat=quat, 
				 name=name, 
				 **kwargs)
		self.attach(*geoms, 
			    *joints, 
			    *bodies, 
			    *sites, 
			    *lights, 
			    *cameras, 
			    *actuators, 
			    *placeholders, 
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
		self._index       = indicies['body']
		indicies['body'] += 1
		return super()._build(parent, world, indicies, **kwargs)

	# KINEMATIC TREES PROPERTIES/METHODS

	@blue.restrict
	@classmethod
	def _from_xml_element(cls, 
			      xml_element: xml.Element, 
			      actuators:   list = []) -> blue.ThingType:
		"""
		This method reconstructs a Body from an xml element.
		
		Parameters
		----------
		xml_element : xml.Element
			The xml element from which a Body is reconstructed.
		
		Returns
		-------
		blue.ThingType
			The reconstructed Body.
		"""
		body = super()._from_xml_element(xml_element)
		for actuator in actuators:
			actuator.body = body
		return body

	# KINEMATIC TREE PROPERTIES

	@property
	def targeting_cameras(self):
		"""
		Targeting Cameras contain :class:`Camera <blueprints.camera.Camera>` that track this Body.
		
		Returns
		-------
		:class:`View <blueprints.utils.view.View>`
		"""
		return blue.View(self._targeting_cameras, name='targeting_cameras', parent=self)


	@targeting_cameras.setter
	@blue.restrict
	def targeting_cameras(self, cameras: blue.CameraType|list[blue.CameraType]|blue.ViewType):
		"""
		Parameters
		----------
		actuators : list[blue.ActuatorType] | blue.ViewType
			Targeting Cameras contain :class:`Camera <blueprints.camera.Camera>` that track this Body.
		"""
		cameras = [cameras] if isinstance(cameras, blue.CameraType) else cameras
		for camera in cameras:
			camera.target = self

	@property
	def targeting_lights(self):
		"""
		Targeting Lights contain :class:`Light <blueprints.light.Light>` that track this Body.
		
		Returns
		-------
		:class:`blueprints.utils.view.View`
		"""
		return blue.View(self._targeting_lights, name='targeting_lights', parent=self)


	@targeting_lights.setter
	@blue.restrict
	def targeting_lights(self, lights: blue.LightType|list[blue.LightType]|blue.ViewType):
		"""
		Parameters
		----------
		actuators : list[blue.ActuatorType] | blue.ViewType
			Targeting Lights contain :class:`Light <blueprints.light.Light>` that track this Body.
		"""
		lights = [lights] if isinstance(lights, blue.LightType) else lights
		for light in lights:
			light.target = self
