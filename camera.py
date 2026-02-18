"""
Cameras provide a rendered image for each time step. A :class:`Camera` can be attached to a :class:`blueprints.body.Body` 
or to the :class:`blueprints.world.World` directly. If a Camera is attached to a :class:`blueprints.body.Body` 
in a :class:`Agent <blueprints.agent.Abent>` its rendered images are contained in the Agents observations. All 
cameras can be viewed when :meth:`blueprints.world.World.view` by setting the ``camera`` in the ``Rendering`` 
section.

Camera Orientation
------------------

Similarly to :class:`blueprints.light.Light` Cameras can either be fixed in orientation locally/globally or 
tracking another Body by setting the :attr:`Camera.mode` and optionally :attr:`Camera.target`. For a detailed 
description see :attr:`Camera.mode`.

.. code-block::
	:caption: Local orientation

	>>> body = blue.Body(cameras=blue.Camera(mode='fixed'))

.. code-block::
	:caption: Global orientation

	>>> body = blue.Body(cameras=blue.Camera(mode='track'))

.. code-block::
	:caption: Tracking Body

	>>> one  = blue.Body(name='one', cameras=blue.Camera(mode='targetbody'))
	>>> two  = blue.Body(name='two')
	>>> root = blue.Body(name='root', bodies=[one, two])
	>>> # Version one:
	>>> root.bodies['one'].cameras.target = root.bodies['two']
	>>> # Version two:
	>>> root.bodies['two'].targeting_cameras = root.bodies['one'].cameras

.. code-block:: mxml
	:caption: XML example
	
	<body name="root">
	    <body name="one">
	        <camera mode="targetbody" name="unnamed_camera" target="two" />
	    </body>
	    <body name="two" />
	</body>

Using the ``target`` attribute for a Camera results in continous tracking of the target Thing. 
If instead a Camera should be initially oriented to center an object, without tracking it during 
the simulation, the :meth:`Camera.looking` method can be used to align the camera to a position or 
direction.

.. code-block::
	:caption: Looking in a Direction

	>>> camera = camera.looking(direction=[1, 1, 1])

.. code-block:: mxml
	:caption: Looking in a Direction

	<camera euler="2.3561945 -0.6154797 0.5235988" />


To center a position either a Thing or a global position can be passed.


.. code-block::
	:caption: Looking at Positions/Things

	>>> body = blue.Body(pos=[-1, 0, 1])
	# VERSION 1
	>>> camera = camera.looking(pos=body)
	# VERSION 2
	>>> camera = camera.looking(pos=[-1, 0, 1])

.. code-block:: mxml
	:caption: Looking at Positions/Things

	<camera euler="1.5707964 2.3561945 1.5707964" />


Recording
---------
To record videos with a Camera you can use the methods :meth:`World.start_recording <blueprints.world.World.start_recording>`, 
:meth:`World.step <blueprints.world.World.step>` and :meth:`World.stop_recording <blueprints.world.World.stop_recording>`.
For more details refer to :meth:`start_recording <blueprints.world.World.start_recording>`.

.. code-block::
	:caption: Recording Videos

	>>> world.start_recording(cam_1, cam_2, cam_3, filename='my.gif', loop=0)
	>>> world.step(seconds=5)
	>>> world.stop_recording()
"""

import xml.etree.ElementTree as xml
import numpy as np
import mujoco
from mujoco.glfw import glfw

import blueprints as blue
from blueprints.utils.geometry import RADIANS_TO_DEGREES, DEGREES_TO_RADIANS, TAU
import time



class Camera(blue.CameraType, blue.thing.CyclicalThing, blue.thing.MoveableThing):

	"""
	This class is available through the shortcut :class:`blueprints.Camera <Camera>`.

	Most attribute descriptions are partially taken from `Mujoco <https://mujoco.readthedocs.io/en/latest/XMLreference.html#body-camera>`__.
	
	Attributes
	----------
	fieldofview : int | float, optional
		The angle of field of view in radians.
	ipd : float
		Inter-pupilary distance. This attribute only has an effect during stereoscopic rendering. It 
		specifies the distance between the left and right viewpoints. Each viewpoint is shifted by 
		+/- half of the distance specified here, along the X axis of the camera frame.
	target : blue.Body | None
		The target attribute specifies a Body that is tracked by the camera. Compatible :attr:`mode` values 
		are ('targetbody', 'targetbodycom').
	mode : str
		The possible values are: 
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
	"""
	
	@blue.restrict
	def __init__(self, 
		     pos:         np.ndarray|list[int|float] = [0., 0., 0.],
		     alpha:       int|float                  = TAU/4, 
		     beta:        int|float                  = 0., 
		     gamma:       int|float                  = 0., 
		     mode:        str                        = 'fixed', 
		     resolution:  np.ndarray|list[int]       = [480, 480],
		     fieldofview: int|float                  = TAU/8, 
		     ipd:         int|float                  = 0.068, 
		     fps:         int|float|None             = 30, 
		     name:        str|None                   = None, 
		     x:           int|float|np.int32|np.int64|np.float32|np.float64|None = None, 
		     y:           int|float|np.int32|np.int64|np.float32|np.float64|None = None, 
		     z:           int|float|np.int32|np.int64|np.float32|np.float64|None = None, 
		     quat:        list[int|float]|np.ndarray|None = None, 
		     **kwargs) -> None:
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
		mode : str, optional
			The mode defines who the position and orientation is changed over time. See :attr:`mode` 
			for a detailed description.
		fieldofview : int | float, optional
			The angle of field of view in radians.
		ipd : int | float, optional
			Inter-pupilary distance. This attribute only has an effect during stereoscopic rendering. 
			It specifies the distance between the left and right viewpoints. Each viewpoint is shifted 
			by +/- half of the distance specified here, along the X axis of the camera frame.
		name : str | None, optional
			The user specified name of the Camera. It might change in the case of a naming conflict. 
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
		assert mode in self._MODES
		self.resolution   = resolution
		self.mode         = mode
		self.fieldofview  = float(fieldofview)
		self.ipd          = ipd
		self.fps          = fps
		super().__init__(pos=pos,
				 alpha=alpha, 
				 beta=beta, 
				 gamma=gamma, 
				 quat=quat, 
				 name=name, 
				 x=x, 
				 y=y, 
				 z=z, 
				 **kwargs)

	@property
	def observation(self):
		"""
		The Camera observation is retrieved as the image rendered by the mujoco Renderer using OpenGL.

		Returns
		-------
		np.ndarray
			RGB image using the specified camera settings.
		"""
		if not isinstance(self.root, blue.WorldType):
			raise Exception(f'Camera observations can only be called once they have been build as part of a World.')
		width, height = self.resolution
		#context = mujoco.GLContext(height, width)
		if not hasattr(self, '_renderer'):
			#self._renderer    = mujoco.Renderer(self.root._mj_model, width, height)
			self._renderer    = mujoco.Renderer(self.root._mj_model, height, width)
			self._last_render = float('-inf')
		if self.root._mj_data.time > self._last_render + self.frameinterval:
			self._renderer.update_scene(self.root._mj_data, self.name)
			for geom in self.root.all.geoms:
				if isinstance(geom, blue.HFieldGeomType):
					mujoco.mjr_uploadHField(self.root._mj_model, self._renderer._mjr_context, geom._index)
			self._rgb = self._renderer.render()
			self._last_render = self.root._mj_data.time
			#print()
			#exit()
		#context.free()
		return self._rgb


	@blue.restrict
	def start_recording(self, 
			    filename: str, 
			    fps:      int = 60, 
			    **kwargs):
		"""
		This shorthand method starts the recording of a video. It can only be 
		used if the camera is already attached within a :class:`World <blueprints.world.World>`.

		For details read :meth:`World.start_recording <blueprints.world.World.start_recording>`.
		
		Parameters
		----------
		filename : str
			The name (with a proper file extension indicating the format) of the video.
			If multiple cameras are passed, multiple videos will be created with the name 
			of each camera prefixed to the filename.
		fps : int
			The frames per second
		**kwargs : dict
			Additional keyword arguments that are passed to imageio to write the video.
		"""
		if isinstance(self.root, blue.WorldType):
			self.root.start_recording(self, 
						  filename=filename, 
						  fps=fps, 
						  **kwargs)
		else:
			raise Exception('A camera can only start recording if it is inhabitant of a World.')

	@blue.restrict
	def stop_recording(self):
		"""
		This shorthand method stops the recording of a video. It can only be 
		used if the camera is already attached within a :class:`World <blueprints.world.World>`.

		For details read :meth:`World.start_recording <blueprints.world.World.start_recording>`.
		"""
		if isinstance(self.root, blue.WorldType):
			self.root.stop_recording(self)
		else:
			raise Exception('A camera can only stop recording if it is inhabitant of a World.')


	@blue.restrict
	def looking(self, 
		    pos:       list[int|float]|np.ndarray|blue.MoveableThingType|None = None, 
		    direction: list[int|float]|np.ndarray|None = None, 
		    **kwargs) -> blue.CameraType:
		"""
		The method creates a new camera that is focussed on pos. The Camera orientation is 
		adjusted to center the ``pos`` argument in the image. Alternatively a direction in 
		which the camera should look can be specified.
		This only constraints two degrees of freedom, so the thrid degree is chosen such that 
		the camera records parallel to the ground.

		Parameters
		----------
		pos : list [ int | float] | np.ndarray | blue.MoveableThingType | None
			If set, it can either be a (global) position or a :class:`MoveableThing <blueprints.thing.moveable.MoveableThing>` 
			that will be centered
		direction :  list [ int | float] | np.ndarray | None
			If set, the returned camera will be oriented parallel to the direction vector.

		Returns
		-------
		blue.CameraType
			A new Camera looking into the specified direction or onto the specified position/Thing
		"""
		if pos is None and direction is None:
			raise ValueError(f'Precisely one argument pos or direction has to be set, but both were None.')
		if pos is not None and direction is not None:
			raise ValueError(f'Precisely one argument pos or direction has to be set, but both were given: pos={pos} direction={direction}')
		if pos is not None:
			if isinstance(pos, list):
				pos = np.array(pos, dtype=np.float32)
			head = self.global_pos
			tail = self._get_pos(pos)
			path =  tail - head
		else:
			path = direction
		# BORROWING MATH FROM TUBE
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
			R   = np.stack([R_x,-R_y,-R_z], axis=1)
			# RECONSTRUCT ANGLES
			alpha, beta, gamma = blue.geometry.Rotation.reference_frame_to_euler(R)
		return self.align(alpha=alpha, 
				  beta=beta, 
				  gamma=gamma, 
				  globally=True, 
				  **kwargs)

	# XML METHODS/FUNCTIONS

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
			The builded xml element of the Thing.
		"""
		if self.target is not None:
			return super()._build(parent, world, indicies, target=self.target.name)
		else:
			return super()._build(parent, world, indicies)

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
		The target attribute specifies a Body that is tracked by the camera. Compatible :attr:`mode` 
		values are ('targetbody', 'targetbodycom').
		
		Parameters
		----------
		target : blue.BodyType | None
			A body which is tracked by the camera. The Body must be part of the same kinematic tree as 
			he Camera.
		
		Raises
		------
		ValueError
			If the parent is not part of the same kinematic tree or the camera has no parent an error 
			is raised.
		"""
		if self.parent is None and not self._IGNORE_CHECKS:
			raise ValueError(f'To set the target reference, parent must be set first.')
		if self.parent is not None and self.parent is target:
			raise ValueError(f'Camera target cannot be the same as its parent. Use the modes ("fixed", "track" "trackcom") instead.')
		if self.target is not None:
			self.target.detach(self)
		if target is not None:
			target._targeting_cameras.append(self)
		self._target = target

	# XML PROPERTIES

	@property
	def resolution(self) -> np.ndarray:
		"""
		The resolution is used for :attr:`observation` rendering and :meth:`recording <blueprints.world.World.start_recording>`. 
		Rendering in :meth:`World.view <blueprints.world.World.view>` uses OpenGL defaults.

		Returns
		-------
		np.ndarray
			The resolution of the camera.
		"""
		return self._resolution.copy()


	@resolution.setter
	@blue.restrict
	def resolution(self, resolution: np.ndarray|list[int|float]) -> None:
		if isinstance(resolution, np.ndarray) and resolution.shape != (2,):
			raise ValueError(f'Position attribute resolution must have a dimension of 2 got {resolution.shape} instead.')
		if isinstance(resolution, list) and len(resolution) != 2:
			raise ValueError(f'Position attribute resolution must have a dimension of 2 got {len(resolution)} instead.')
		self._resolution = np.array(resolution, dtype=np.uint16)


	@property
	@blue.restrict
	def ipd(self) -> float:
		"""
		This attribute only has an effect during stereoscopic rendering. 
		It specifies the distance between the left and right viewpoints. 
		Each viewpoint is shifted by :math:`\\pm` half of the distance 
		specified here, along the :math:`X` axis of the camera frame.
		
		Returns
		-------
		float
			Inter-pupilary distance
		"""
		
		return self._ipd


	@ipd.setter
	@blue.restrict
	def ipd(self, ipd: int|float) -> None:
		"""
		Parameters
		----------
		ipd : float | int
			Inter-pupilary distance
		"""
		self._ipd = float(ipd)


	@property
	@blue.restrict
	def fieldofview(self) -> float:
		"""
		Vertical field-of-view of the camera.

		Returns
		-------
		float
			Field of view
		"""
		return self._fieldofview


	@fieldofview.setter
	@blue.restrict
	def fieldofview(self, fieldofview: int|float) -> None:
		"""
		Parameters
		----------
		fieldofview : float | int
			Field of view
		"""
		self._fieldofview = float(fieldofview)


	@property
	@blue.restrict
	def fps(self) -> float|None:
		"""
		Frames per second
		
		Returns
		-------
		float
		"""
		return self._fps


	@fps.setter
	@blue.restrict
	def fps(self, fps: int|float|None) -> None:
		"""
		If `fps` is set to `None` a new image will be rendered on each forward step of the physics engine.

		Parameters
		----------
		fps : float | int | None
			Frames per second
		"""
		if fps is not None and fps <= 0.:
			raise ValueError('Frames per second can only be positive.')
		self._fps = float(fps) if fps is not None else None
	
	# DERIVATIONS

	@property
	def fovy(self) -> float:
		"""
		Vertical field-of-view of the camera.
		
		Returns
		-------
		float
			field of view in degrees
		"""
		return self.fieldofview * RADIANS_TO_DEGREES


	@fovy.setter
	@blue.restrict
	def fovy(self, 
		 fovy: float):
		"""
		Parameters
		----------
		fovy : float
			field of view in degrees
		"""
		self.fieldofview = fovy * DEGREES_TO_RADIANS


	@property
	def frameinterval(self) -> float:
		"""
		The delay between each rendering in seconds
		
		Returns
		-------
		float
		"""
		if self.fps is None:
			return 0.
		else:
			return 1/self.fps
