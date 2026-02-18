"""
Lights can be used to illuminate your models. They have a position and an orientation which can be set via 
the standard orientation attributes of :class:`blueprints.thing.moveable.MoveableThing`. Mujoco uses the 
:attr:`Light.dir` attribute internally, so blueprint synchronizes all changes in ``euler``, ``alpha``, ``beta``, 
``gamma`` with ``dir`` internally.

Similarly to :class:`blueprints.camera.Camera` Lights can either be fixed in orientation locally/globally or 
tracking another Body by setting the :attr:`Light.mode` and optionally :attr:`Light.target`. For a detailed 
description see :attr:`Light.mode`.

.. code-block::
	:caption: Local orientation

	>>> body = blue.Body(lights=blue.Light(mode='fixed'))

.. code-block::
	:caption: Global orientation

	>>> body = blue.Body(lights=blue.Light(mode='track'))

.. code-block::
	:caption: Tracking Body

	>>> one  = blue.Body(name='one', lights=blue.Light(mode='targetbody'))
	>>> two  = blue.Body(name='two')
	>>> root = blue.Body(name='root', bodies=[one, two])
	>>> # Version one:
	>>> root.bodies['one'].lights.target = root.bodies['two']
	>>> # Version two:
	>>> root.bodies['two'].targeting_lights = root.bodies['one'].lights

.. code-block:: mxml
	:caption: XML example
	
	<body name="root">
	    <body name="one">
	        <light mode="targetbody" name="unnamed_light" cutoff="90.0" target="two" />
	    </body>
	    <body name="two" />
	</body>
"""

import numpy as np
import xml.etree.ElementTree as xml
import blueprints as blue
from blueprints.utils.geometry import TAU, PI



class Light(blue.LightType, blue.thing.CyclicalThing, blue.MoveableThing):

	"""
	This class is available through the shortcut :class:`blueprints.Light <Light>`.

	Most attribute descriptions are partially taken from `Mujoco <https://mujoco.readthedocs.io/en/latest/XMLreference.html#body-light>`__.
	"""
	
	@blue.restrict
	def __init__(self, 
		     pos:         np.ndarray|list[int|float] = [0., 0., 0.], 
		     dir:         np.ndarray|list[int|float] = [0., 0.,-1.], 
		     attenuation: np.ndarray|list[int|float] = [1., 0., 0.],
		     ambient:     np.ndarray|list[int|float] = [0., 0., 0.], 
		     diffuse:     np.ndarray|list[int|float] = [.7, .7, .7], 
		     specular:    np.ndarray|list[int|float] = [.3, .3, .3], 
		     mode:        str                        = 'fixed', 
		     directional: bool                       =  False, 
		     castshadow:  bool                       =  True, 
		     active:      bool                       =  True, 
		     cutoff:      float|int                  =  TAU, 
		     exponent:    float|int                  =  10., 
		     name:        str|None                   =  None, 
		     x:           int|float|np.int32|np.int64|np.float32|np.float64|None = None, 
		     y:           int|float|np.int32|np.int64|np.float32|np.float64|None = None, 
		     z:           int|float|np.int32|np.int64|np.float32|np.float64|None = None, 
		     **kwargs):
		"""
		Parameters
		----------
		pos : np.ndarray | list[int | float], optional
			Represents the position of the Light. Changing this attribute also changes the properties 
			:attr:`x`, :attr:`y` and :attr:`z`.
		dir : np.ndarray | list[int | float], optional
			Direction of the Light.
		attenuation : np.ndarray | list[int | float], optional
			These are the constant, linear and quadratic attenuation coefficients in OpenGL. The default 
			corresponds to no attenuation. See the OpenGL documentation for more information on this and 
			all other OpenGL-related properties.
		ambient : np.ndarray | list[int | float], optional
			The ambient color of the Light.
		diffuse : np.ndarray | list[int | float], optional
			The diffuse color of the Light.
		specular : np.ndarray | list[int | float], optional
			The specular color of the Light.
		mode : str, optional
			The mode defines who the position and orientation is changed over time. See :attr:`mode` 
			for a detailed description.
		directional : bool, optional
			The light is directional if this attribute is True, otherwise it is a spotlight.
		castshadow : bool, optional
			If this attribute is True the Light will cast shadows. More precisely, the geoms 
			illuminated by the light will cast shadows, however this is a property of Lights rather 
			than geoms. Since each shadow-casting light causes one extra rendering pass through all 
			geoms, this attribute should be used with caution.
		active : bool, optional
			The Light is active if this attribute is True. This can be used at runtime to turn Lights 
			on and off.
		cutoff : float, optional
			Cutoff angle for spotlights in radians.
		exponent : float, optional
			Exponent for spotlights. This setting controls the softness of the spotlight cutoff.
		name : str | None, optional
			The user specified name of the Camera. It might change in the case of a naming conflict. 
		x : int | float |np.int32 | np.int64 | np.float32 | np.float64 | None, optional
			If `pos` is not specified, this argument sets the X position coordinate.
		y : int | float |np.int32 | np.int64 | np.float32 | np.float64 | None, optional
			If `pos` is not specified, this argument sets the Y position coordinate.
		z : int | float |np.int32 | np.int64 | np.float32 | np.float64 | None, optional
			If `pos` is not specified, this argument sets the Z position coordinate.
		**kwargs
			Keyword arguments are passed to ``super().__init__``.
		"""
		assert mode in self._MODES
		if 'euler' in kwargs:
			alpha, beta, gamma = kwargs['euler']
			R = blue.geometry.Rotation.E_rot(alpha, beta, gamma)
			dir = R.T @ dir
		# ASSIGN ATTRIBUTES
		self.dir         = dir
		self.attenuation = attenuation
		self.ambient     = ambient
		self.diffuse     = diffuse
		self.specular    = specular
		self.mode        = mode
		self.directional = directional
		self.castshadow  = castshadow
		self.active      = active
		self.cutoff      = cutoff
		self.exponent    = exponent
		super().__init__(pos=pos, 
				 name=name, 
				 x=x, 
				 y=y, 
				 z=z, 
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
		
		Returns
		-------
		xml.etree.ElementTree.Element
			The builded xml element of the Light.
		"""
		self._index        = indicies['light']
		indicies['light'] += 1
		cutoff = self.cutoff * blue.geometry.RADIANS_TO_DEGREES
		if self.target is not None:
			return super()._build(parent, world, indicies, cutoff=cutoff, target=self.target.name)
		else:
			return super()._build(parent, world, indicies, cutoff=cutoff)


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
			The reconstructed Light.
		"""
		init_args, post_args, rest_args = cls._xml_element_args(xml_element)
		init_args['copy'] = False
		if 'cutoff' in init_args:
			init_args['cutoff'] = blue.geometry.DEGREES_TO_RADIANS * init_args['cutoff']
		obj = object.__new__(cls)
		obj.__init__(**init_args)
		for key, val in post_args.items():
			setattr(obj, key, val)
		return obj

	# KINEMATIC TREE PROPERTIES

	@property
	def target(self) -> blue.BodyType|None:
		"""
		The target attribute specifies a Body that is tracked by the camera. Compatible :attr:`mode` 
		values are ('targetbody', 'targetbodycom').
		
		Returns
		-------
		blue.BodyType | None
		"""
		if hasattr(self, '_target'):
			return self._target
		else:
			return None


	@target.setter
	@blue.restrict
	def target(self, target: blue.BodyType|None) -> None:
		"""
		The target attribute specifies a Body that is tracked by the Light. Compatible :attr:`mode` 
		values are ('targetbody', 'targetbodycom').
		
		Parameters
		----------
		target : blue.BodyType | None
			A body which is tracked by the Light. The Body must be part of the same kinematic tree 
			as the Light.
		
		Raises
		------
		ValueError
			If the parent is not part of the same kinematic tree or the Light has no parent an error 
			is raised.
		"""
		if self.parent is None and not self._IGNORE_CHECKS:
			raise ValueError(f'To set the target reference, parent must be set first.')
		if self.parent is not None and self.parent is target:
			raise ValueError(f'Light target cannot be the same as its parent. Use the modes ("fixed", "track" "trackcom") instead.')
		if self.target is not None:
			self.target.detach(self)
		if target is not None:
			target._targeting_lights.append(self)
		self._target = target

	# MUJOCO PROPERTIES

	@property
	def mode(self) -> str:
		"""
		This argument works identically to :attr:`blueprints.camera.Camera.mode`. Possible values are:

		 * ``'fixed'``: Position and orientation will change w.r.t. the parents frame of reference.
		 * ``'track'``:  Position is at a constant offset from the parent in world coordinates, while the 
		   camera orientation is constant in world coordinates. 
		 * ``'trackcom'``:  is similar to ``'track'`` but the constant spatial offset is defined relative 
		   to the center of mass of the kinematic subtree starting at the parent Body. This can be used to 
		   keep an entire mechanism in view. Note that the subtree center of mass for the World is the 
		   center of mass of the entire model.
		 * ``'targetbody'``: the Camera position is fixed in the parent Body, while the Camera orientation 
		   is adjusted so that it always points towards the targeted Body (which is specified with the 
		   target attribute)
		 * ``'targetbodycom'``: is the same as ``'targetbody'`` but the Camera is oriented towards the 
		   center of mass of the subtree starting at the target Cody.

		Returns
		-------
		str
			When setting this value manually, it must be one of those specified above, otherwise an error is raised.
		"""
		return self._mode


	@mode.setter
	@blue.restrict
	def mode(self, mode: str) -> None:
		"""
		This argument works identically to :attr:`blueprints.camera.Camera.mode`. Possible values are: 
		 * ``'fixed'``: Position and orientation will change w.r.t. the parents frame of reference.
		 * ``'track'``:  Position is at a constant offset from the parent in world coordinates, while the 
		   camera orientation is constant in world coordinates. 
		 * ``'trackcom'``:  is similar to ``'track'`` but the constant spatial offset is defined relative 
		   to the center of mass of the kinematic subtree starting at the parent Body. This can be used to 
		   keep an entire mechanism in view. Note that the subtree center of mass for the World is the 
		   center of mass of the entire model.
		 * ``'targetbody'``: the Camera position is fixed in the parent Body, while the Camera orientation 
		   is adjusted so that it always points towards the targeted Body (which is specified with the 
		   target attribute)
		 * ``'targetbodycom'``: is the same as ``'targetbody'`` but the Camera is oriented towards the 
		   center of mass of the subtree starting at the target Cody.
		
		
		
		Parameters
		----------
		mode : str
			The attribute must be one of those specified above.
		
		Raises
		------
		ValueError
			If ``mode`` is not a valid value an error is raised.
		"""
		if mode not in self._MODES:
			raise ValueError(f'The allowed values for mode are {self._MODES}, got {mode} instead.')
		self._mode = mode


	@property
	def dir(self) -> np.ndarray:
		"""
		Direction of the light.
		
		Returns
		-------
		np.ndarray
		"""
		return self.rotation_matrix @ self._dir.copy()


	@dir.setter
	@blue.restrict
	def dir(self, dir: np.ndarray|list[int|float]) -> None:
		"""
		Parameters
		----------
		dir : np.ndarray | list[int | float]
			Direction of the light.
		
		Raises
		------
		Exception
			If the shape of the dir argument is not (3,) and error is raised.
		"""
		if isinstance(dir, np.ndarray) and dir.shape != (3,):
			raise Exception(f'Position attribute dir must have a dimension of 3 got {dir.shape} instead.')
		if isinstance(dir, list) and len(dir) != 3:
			raise Exception(f'Position attribute dir must have a dimension of 3 got {len(dir)} instead.')
		self._dir = np.array(dir, dtype=np.float32)


	@property
	def attenuation(self) -> np.ndarray:
		"""
		These are the constant, linear and quadratic attenuation coefficients in OpenGL. The 
		default corresponds to no attenuation. See the OpenGL documentation for more information 
		on this and all other OpenGL-related properties.
		
		Returns
		-------
		np.ndarray
		"""
		return self._attenuation.copy()


	@attenuation.setter
	@blue.restrict
	def attenuation(self, attenuation: np.ndarray|list[int|float]) -> None:
		"""
		Parameters
		----------
		attenuation : np.ndarray | list[int | float]
			These are the constant, linear and quadratic attenuation coefficients in OpenGL. The 
			default corresponds to no attenuation. See the OpenGL documentation for more information 
			on this and all other OpenGL-related properties.
		
		Raises
		------
		Exception
			If the shape of the dir argument is not (3,) and error is raised.
		"""
		if isinstance(attenuation, np.ndarray) and attenuation.shape != (3,):
			raise Exception(f'Position attribute attenuation must have a dimension of 3 got {attenuation.shape} instead.')
		if isinstance(attenuation, list) and len(attenuation) != 3:
			raise Exception(f'Position attribute attenuation must have a dimension of 3 got {len(attenuation)} instead.')
		self._attenuation = np.array(attenuation, dtype=np.float32)


	@property
	def ambient(self) -> np.ndarray:
		"""
		The ambient color of the light (RGB).
		
		Returns
		-------
		np.ndarray
		"""
		return self._ambient.copy()


	@ambient.setter
	@blue.restrict
	def ambient(self, ambient: np.ndarray|list[int|float]) -> None:
		"""
		Parameters
		----------
		ambient : np.ndarray | list[int | float]
			The ambient color of the light (RGB).
		
		Raises
		------
		Exception
		   If the shape of the dir argument is not (3,) and error is raised.
		"""
		if isinstance(ambient, np.ndarray) and ambient.shape != (3,):
			raise Exception(f'Position attribute ambient must have a dimension of 3 got {ambient.shape} instead.')
		if isinstance(ambient, list) and len(ambient) != 3:
			raise Exception(f'Position attribute ambient must have a dimension of 3 got {len(ambient)} instead.')
		self._ambient = np.array(ambient, dtype=np.float32)


	@property
	def diffuse(self) -> np.ndarray:
		"""
		The diffuse color of the light (RGB).
		
		Returns
		-------
		np.ndarray
		"""
		return self._diffuse.copy()


	@diffuse.setter
	@blue.restrict
	def diffuse(self, diffuse: np.ndarray|list[int|float]) -> None:
		"""
		Parameters
		----------
		diffuse : np.ndarray | list[int | float]
			The diffuse color of the light (RGB).
		
		Raises
		------
		Exception
			If the shape of the dir argument is not (3,) and error is raised.
		"""
		if isinstance(diffuse, np.ndarray) and diffuse.shape != (3,):
			raise Exception(f'Position attribute diffuse must have a dimension of 3 got {diffuse.shape} instead.')
		if isinstance(diffuse, list) and len(diffuse) != 3:
			raise Exception(f'Position attribute diffuse must have a dimension of 3 got {len(diffuse)} instead.')
		self._diffuse = np.array(diffuse, dtype=np.float32)


	@property
	def specular(self) -> np.ndarray:
		"""
		The specular color of the light (RGB).
		
		Returns
		-------
		np.ndarray
		"""
		return self._specular.copy()


	@specular.setter
	@blue.restrict
	def specular(self, specular: np.ndarray|list[int|float]) -> None:
		"""
		Parameters
		----------
		specular : np.ndarray | list[int | float]
			The specular color of the light (RGB).
		
		Raises
		------
		Exception
			If the shape of the dir argument is not (3,) and error is raised.
		"""
		if isinstance(specular, np.ndarray) and specular.shape != (3,):
			raise Exception(f'Position attribute specular must have a dimension of 3 got {specular.shape} instead.')
		if isinstance(specular, list) and len(specular) != 3:
			raise Exception(f'Position attribute specular must have a dimension of 3 got {len(specular)} instead.')
		self._specular = np.array(specular, dtype=np.float32)


	@property
	def directional(self) -> bool:
		"""
		The light is directional if this attribute is “true”, otherwise it is a spotlight.
		
		Returns
		-------
		bool
		"""
		return self._directional


	@directional.setter
	@blue.restrict
	def directional(self, directional: bool) -> None:
		"""
		Parameters
		----------
		directional : bool
			The light is directional if this attribute is “true”, otherwise it is a spotlight.
		"""
		self._directional = directional


	@property
	def castshadow(self) -> bool:
		"""
		If this attribute is True the Light will cast shadows. More precisely, the geoms 
		illuminated by the light will cast shadows, however this is a property of Lights rather 
		than geoms. Since each shadow-casting light causes one extra rendering pass through all 
		geoms, this attribute should be used with caution.
		
		Returns
		-------
		bool
		"""
		return self._castshadow


	@castshadow.setter
	@blue.restrict
	def castshadow(self, castshadow: bool) -> None:
		"""
		Parameters
		----------
		castshadow : bool
			If this attribute is True the Light will cast shadows. More precisely, the geoms 
			illuminated by the light will cast shadows, however this is a property of Lights rather 
			than geoms. Since each shadow-casting light causes one extra rendering pass through all 
			geoms, this attribute should be used with caution.
		"""
		self._castshadow = castshadow


	@property
	def active(self) -> bool:
		"""
		The light is active if this attribute is “true”. This can be used at runtime to turn 
		lights on and off.
		
		Returns
		-------
		bool
		"""
		return self._active


	@active.setter
	@blue.restrict
	def active(self, active: bool) -> None:
		"""
		Parameters
		----------
		active : bool
			The light is active if this attribute is “true”. This can be used at runtime to turn 
			lights on and off.
		"""
		self._active = active


	@property
	def cutoff(self) -> float:
		"""
		Cutoff angle for spotlights in radians.
		
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
		cutoff : float
			Cutoff angle for spotlights in radians.
		"""
		self._cutoff = float(cutoff)


	@property
	def exponent(self) -> float:
		"""
		Exponent for spotlights. This setting controls the softness of the spotlight cutoff.
		
		Returns
		-------
		float
		"""
		return self._exponent


	@exponent.setter
	@blue.restrict
	def exponent(self, exponent: float|int) -> None:
		"""
		Parameters
		----------
		exponent : float
			Exponent for spotlights. This setting controls the softness of the spotlight cutoff.
		"""
		self._exponent = float(exponent)
