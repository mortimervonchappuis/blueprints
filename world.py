"""
The :class:`World` class hosts all Things in a Simulation. It handles the access to Mujoco by implementing a 
method, that converts the Constructed World to an xml string and a method used to launch a Mujoco viewer, that 
is especially useful during model construction for debugging. You can add and remove Things from the World with 
the same methods as in :class:`NodeThing <blueprints.thing.node.NodeThing>` i.e. :meth:`attach <blueprints.thing.node.NodeThing.attach>` 
and :meth:`detach <blueprints.thing.node.NodeThing.detach>`.

.. code-block:: python
	:caption: World attachment/detachment

	>>> world = blue.World()
	>>> teapot = blue.geoms.Mesh(filename='utahteapot.stl', pos=[0, 0, 1])
	>>> red_teapot = teapot.copy(color='red')
	>>> green_teapot = teapot.copy(color='green')
	>>> world.attach(red_teapot, green_teapot, copy=False)
	>>> world.geoms.color
	[Red[#FF00], Green[#0FF0]]
	>>> world.detach(red_teapot)
	>>> world.geoms.color
	[Green[#0FF0]]

The :meth:`view <World.view>` method can be used. The perspective of the default camera is set such that all 
Things attached to the :class:`World` are in view.

.. code-block:: python
	:caption: World view

	>>> world.view()

Things that are attached with the ``background`` flag argument set to ``True`` will not be used to center the 
perspective of the camera.

.. code-block:: python
	:caption: World background

	>>> plane = blue.geoms.Plane()
	>>> world.attach(plane, background=True)
	>>> world.view()

"""

import xml.etree.ElementTree as xml
import numpy as np
import os
import sys
import time
import tqdm
import imageio
import mujoco
import mujoco.viewer
from mujoco.glfw import glfw
import blueprints as blue

from collections import defaultdict
from itertools import chain



class World(blue.WorldType, blue.thing.NodeThing):

	r"""
	This class is available through the shortcut :class:`blueprints.World <World>`.

	Most attribute descriptions are partially taken from `Mujoco <https://mujoco.readthedocs.io/en/latest/XMLreference.html#mjcf-reference>`__.
	
	Attributes
	----------
	angle : str
		Either ``'radian'`` or ``'degree'`` (avoid if possible).
	autolimits : bool
		Indicating whether limits are used or not.
	contact : bool
		Indicating whether collisions and contacts are ignored in the simulations physics.
	gravity : bool
		Indicating whether gravity exists.
	integrator : str
		The name of the integrator use, see :attr:`integrator` for a detailed description.
	viscosity : float
		The viscosity of the Worlds Things.
	"""

	@blue.restrict
	def __init__(self, 
		     name:       str            = 'anonymous_model', 
		     autolimits: bool           = True, 
		     viscosity:  float|int|None = 0.,
		     timestep:   float|int|None = 0.002,
		     integrator: str            = 'implicit',
		     cone:       str            = 'pyramidal', 
		     gravity:    bool           = True, 
		     contact:    bool           = True, 
		     angle:      str            = 'radian', 
		     texture:    blue.SkyboxTextureType|None = None, 
		     **kwargs):
		r"""
		Parameters
		----------
		name : str, optional
			The user specified name for the Actuator. In the case of a naming conflict the name will 
			be altered by an enumeration scheme.
		autolimits : bool, optional
			This attribute affects the behavior of attributes such as ``'limited'`` (on <body-joint> 
			``'forcelimited'``, ``'ctrllimited'``, and ``'actlimited'`` (on <actuator>). If ``True``, 
			these attributes are unnecessary and their value will be inferred from the presence of their 
			corresponding ``'range'`` attribute. If ``False``, no such inference will happen: For a Joint 
			to be limited, both limited=``True`` and range=``min max`` must be specified. In this mode, 
			it is an error to specify a range without a limit. 
		timestep : float | int | None, optional
			Simulation time step in seconds. This is the single most important parameter affecting 
			the speed-accuracy trade-off which is inherent in every physics simulation. Smaller values 
			result in better accuracy and stability. To achieve real-time performance, the time step 
			must be larger than the CPU time per step (or 4 times larger when using the RK4 integrator). 
			The CPU time is measured with internal timers. It should be monitored when adjusting the 
			time step.
		viscosity : float | int | None, optional
			Viscosity of the medium. This parameter is used to simulate viscous forces, which scale 
			linearly with velocity. In SI units the viscosity of air is around 0.00002 while the 
			viscosity of water is around 0.0009 depending on temperature. Setting viscosity to 0 
			disables viscous forces. Note that the default Euler integrator handles damping in the 
			:class:`Joints <blueprints.joint.BaseJoint>` implicitly – which improves stability and 
			accuracy. It does not presently do this with body viscosity. Therefore, if the goal is 
			merely to create a damped simulation (as opposed to modelling the specific effects of 
			viscosity), we recommend using joint damping rather than body viscosity, or switching to 
			the implicit or implicitfast :attr:`integrator`.
		cone : str, optional
			The type of contact friction cone. Elliptic cones are a better model of the physical reality, but pyramidal cones sometimes make the solver faster and more robust.
		integrator : str, optional
			This attribute selects the numerical integrator to be used. Currently the available 
			integrators are the semi-implicit Euler method ``'Euler'``, the fixed-step 4-th order 
			Runge Kutta method ``'RK4'``, the Implicit-in-velocity Euler method ``'implicit'``, and  
			``'implicitfast'``, which drops the Coriolis and centrifugal terms.
		gravity : bool, optional
			A flag indicating whether gravity exists. Be default the `gravitational constant <https://en.wikipedia.org/wiki/Gravitational_constant>`__. 
			is set to :math:`6.674 \times 10^{-11} N \frac{m^2}{kg^2}`.
		contact : bool, optional
			If ``False``, collisions and contacts are ignored in the simulations physics.
		angle : str, optional
			The angle measure either ``'radian'`` or ``'degree'``. 
			.. warning::
			
			    At the moment 'degree' is not supported for any manipulations. It is recommended to 
			    only use 'degree' if no orientation manipulation is performed on the model! To convert 
			    from degrees to radians use the multiplier :attr:`DEGREES_TO_RADIANS <blueprints.utils.geometry.DEGREES_TO_RADIANS>`
		**kwargs
			Keyword arguments are passed to ``super().__init__``.
		"""
		self.angle       = angle
		self.gravity     = gravity
		self.timestep    = float(timestep)
		self.viscosity   = float(viscosity)
		self.integrator  = integrator
		self.cone        = cone
		self.contact     = contact
		self.autolimits  = autolimits
		self.texture     = texture
		self._agents     = []
		self._geoms      = []
		self._bodies     = []
		self._sites      = []
		self._lights     = []
		self._cameras    = []
		self._background = []
		self._CHILDREN   = {'lights':  {'type':     blue.LightType, 
						'children': self._lights}, 
				    'geoms':   {'type':     blue.GeomType, 
						'children': self._geoms}, 
				    'sites':   {'type':     blue.SiteType, 
						'children': self._sites}, 
				    'cameras': {'type':     blue.CameraType, 
						'children': self._cameras}, 
				    'agents':  {'type':     blue.AgentType, 
				    		'children': self._agents}, 
				    'bodies':  {'type':     blue.BodyType, 
						'children': self._bodies}}
		self._name_manager = blue.naming.NameManager(self)
		self._built        = False
		self._viewer       = None
		self._recordings   = dict()
		super().__init__(name=name, 
				 **kwargs)

	# KINEMATIC TREE PROPERTIES/METHODS

	@blue.restrict
	def attach(self, 
		   *items:      list[blue.ThingType|blue.LatticeType|blue.ViewType], 
		   copy:       bool = False, 
		   background: bool = False) -> None:
		"""
		This method attaches *items to the parent Thing. If copy is set true, a copy of the items is created. 
		If copy is set to false, the kinematic graph might no longer be a tree resulting in infinite loops 
		or naming conflicts if the same item is included in the tree twice. All items of a specific type 
		can be accessed via the types attribute.
		
		Parameters
		----------
		*items : list[blue.ThingType]
			All items that are to be attached.
		copy : bool, optional
			A flag indicating whether a copy of the items should be attached.
		background : bool, optional
			A flag indicating whether the attached Things belong to the Worlds background.

		Raises
		------
		TypeError
			If an item is not a valid child an error is raised.
		"""
		views   =                   list(chain( *filter(lambda x:     isinstance(x, blue.ViewType),                     items)))
		lattice =                   list(chain( *filter(lambda x:     isinstance(x, blue.LatticeType),                  items)))
		items   = views + lattice + list(        filter(lambda x: not isinstance(x, (blue.ViewType, blue.LatticeType)), items))
		if copy:
			items = [item.copy() for item in items]
		super().attach(*items, copy=False)
		if background:
			self._background.extend(items)


	@blue.restrict
	def detach(self, *items: list[blue.ThingType]) -> None:
		"""
		This method is used to detach Things from the kinematic tree. They are afterwards 
		no longer present in this Worlds children attributes and their ``parent`` attribute 
		will be set to ``None``. 
		
		Parameters
		----------
		*items
			Things that are no longer children of this World can be passed, as long as their 
			type is a valid type for children of :class:`World`.
		
		Raises
		------
		TypeError
			If the arguments to this function are not valid children types an error is raised.
		"""
		super().detach(*items)
		for item in items:
			if item in self._background:
				self._background.remove(item)

	# XML PROPERTIES/METHODS

	@blue.restrict
	def build(self) -> None:
		"""
		This method constructs the xml structure of the kinematic tree. It is called before the World 
		is converted to an xml string if any modification to it was done since the last build. It is 
		not necessary for a user to call this method since it is implicitly called every time the build 
		needs to be updated.
		"""
		if self._built:
			return
		# REGISTERING NAMES
		self._name_manager.register()
		# BUILD DIRECTORY
		pathname   = os.path.dirname(sys.argv[0])
		self._path = os.path.abspath(pathname)
		# BUILD SIZE AND CENTER
		min_pos, max_pos    = self._location_range
		self._center        = (min_pos + max_pos)/2
		self._size          = np.max(max_pos - min_pos)/2
		# BUILD TOP LEVEL XML
		self._xml_root      = xml.Element('mujoco', model=self.name)
		self._xml_compiler  = xml.SubElement(self._xml_root, 
							 'compiler', 
							 angle=self.angle, 
							 autolimits=str(self.autolimits).lower(), 
							 #inertiafromgeom='true', 
							 coordinate='local')
		option = xml.SubElement(self._xml_root, 
					'option', 
					timestep=str(self.timestep), 
					viscosity=str(self.viscosity), 
					integrator=self.integrator, 
					cone=self.cone)
		#statistic = xml.SubElement(self._xml_root, 
		#			'statistic', 
		#			 meansize=".05"
		#			)
		flag   = xml.SubElement(option, 
					'flag', 
					gravity='disable' if not self.gravity else 'enable', 
					contact='disable' if not self.contact else 'enable')
		self._xml_asset     = xml.SubElement(self._xml_root, 'asset')
		self._xml_sensor    = xml.SubElement(self._xml_root, 'sensor')
		self._xml_actuator  = xml.SubElement(self._xml_root, 'actuator')
		self._xml_tendon    = xml.SubElement(self._xml_root, 'tendon')
		
		if self.all.cameras:
			max_width, max_height = np.max(self.all.cameras.resolution, axis=0)
		else:
			max_width, max_height = 480, 480
		self._gl_context = mujoco.GLContext(max_height, max_width)
		
		self._xml_visual    = xml.SubElement(self._xml_root, 'visual')
		self._xml_global    = xml.SubElement(self._xml_visual, 'global', offwidth=str(max_width), offheight=str(max_height))
		# BUILD WORLDBODY XML
		indicies = {'body':       1, 
			    'geom':       0, 
			    'jnt':        0, 
			    'light':      0, 
			    'mesh':       0, 
			    'site':       0, 
			    'sensors':    0, 
			    'activation': 0, 
			    'force':      0, 
			    'hfield':     0}
		self._xml_worldbody = xml.SubElement(self._xml_root, 'worldbody')
		self._build_children(parent=self._xml_worldbody, 
				     world=self, 
				     indicies=indicies)
		if self.texture is not None:
			self.texture._build(self, self, indicies)
		# CLEAN UP EMPTY TOP LEVEL CONTAINERS
		if len(self._xml_asset) == 0:
			assets = self._xml_root.find('asset')
			self._xml_root.remove(assets)
		if len(self._xml_sensor) == 0:
			sensors = self._xml_root.find('sensor')
			self._xml_root.remove(sensors)
		if len(self._xml_actuator) == 0:
			actuators = self._xml_root.find('actuator')
			self._xml_root.remove(actuators)
		# SET INDENTS
		xml.indent(self._xml_root, '\t', 0)
		self._built = True
		self.reset()


	def reset(self):
		"""
		This method resets the Mujoco simulation data and initializes a single first step 
		to make all runtime data accessable.
		"""
		if self._built:
			xml_string     = self.to_xml_string()
			self._mj_model = mujoco.MjModel.from_xml_string(xml_string)
			self._mj_data  = mujoco.MjData(self._mj_model)
			mujoco.mj_forward(self._mj_model, self._mj_data)
		else:
			raise Exception('World can only be reset after it has been built.')
		


	@blue.restrict
	def unbuild(self) -> None:
		"""
		Unbuilds the World to anable alterations. This discards the current Mujoco simulation.
		"""
		if self._built:
			self._built = False
			for tendon in self.descendants['tendons']['descendants']:
				tendon._built = False
			self._name_manager.unregister()
			self._gl_context.free()
			del self._mj_model, self._mj_data, self._gl_context
			self.stop_recording()


	@blue.restrict
	def view(self, callback=lambda: None) -> None:
		"""
		Launches an interactive window of the Mujoco simulation.

		Parameters
		----------
		callback : function
			The callback function is called after each timestep. The user can use this 
			function to update World properties during the simulation.
		"""
		self.build()
		xml_string = self.to_xml_string()
#		xml_string = """
#<mujoco model="humanoid">
#    <compiler angle="degree" inertiafromgeom="true"/>
#    <default>
#        <joint armature="1" damping="1" limited="true"/>
#        <geom conaffinity="1" condim="1" contype="1" margin="0.001" material="geom" rgba="0.8 0.6 .4 1"/>
#        <motor ctrllimited="true" ctrlrange="-.4 .4"/>
#    </default>
#    <option integrator="RK4" iterations="50" solver="PGS" timestep="0.003">
#        <!-- <flags solverstat="enable" energy="enable"/>-->
#    </option>
#    <size nkey="5" nuser_geom="1"/>
#    <visual>
#        <map fogend="5" fogstart="3"/>
#    </visual>
#    <asset>
#        <texture builtin="gradient" height="100" rgb1=".4 .5 .6" rgb2="0 0 0" type="skybox" width="100"/>
#        <!-- <texture builtin="gradient" height="100" rgb1="1 1 1" rgb2="0 0 0" type="skybox" width="100"/>-->
#        <texture builtin="flat" height="1278" mark="cross" markrgb="1 1 1" name="texgeom" random="0.01" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" type="cube" width="127"/>
#        <texture builtin="checker" height="100" name="texplane" rgb1="0 0 0" rgb2="0.8 0.8 0.8" type="2d" width="100"/>
#        <material name="MatPlane" reflectance="0.5" shininess="1" specular="1" texrepeat="60 60" texture="texplane"/>
#        <material name="geom" texture="texgeom" texuniform="true"/>
#    </asset>
#    <worldbody>
#        <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>
#        <geom condim="3" friction="1 .1 .1" material="MatPlane" name="floor" pos="0 0 0" rgba="0.8 0.9 0.8 1" size="20 20 0.125" type="plane"/>
#        <!-- <geom condim="3" material="MatPlane" name="floor" pos="0 0 0" size="10 10 0.125" type="plane"/>-->
#        <body name="torso" pos="0 0 1.4">
#            <camera name="track" mode="trackcom" pos="0 -4 0" xyaxes="1 0 0 0 0 1"/>
#            <joint armature="0" damping="0" limited="false" name="root" pos="0 0 0" stiffness="0" type="free"/>
#            <geom fromto="0 -.07 0 0 .07 0" name="torso1" size="0.07" type="capsule"/>
#            <geom name="head" pos="0 0 .19" size=".09" type="sphere" user="258"/>
#            <geom fromto="-.01 -.06 -.12 -.01 .06 -.12" name="uwaist" size="0.06" type="capsule"/>
#            <body name="lwaist" pos="-.01 0 -0.260" quat="1.000 0 -0.002 0">
#                <geom fromto="0 -.06 0 0 .06 0" name="lwaist" size="0.06" type="capsule"/>
#                <joint armature="0.02" axis="0 0 1" damping="5" name="abdomen_z" pos="0 0 0.065" range="-45 45" stiffness="20" type="hinge"/>
#                <joint armature="0.02" axis="0 1 0" damping="5" name="abdomen_y" pos="0 0 0.065" range="-75 30" stiffness="10" type="hinge"/>
#                <body name="pelvis" pos="0 0 -0.165" quat="1.000 0 -0.002 0">
#                    <joint armature="0.02" axis="1 0 0" damping="5" name="abdomen_x" pos="0 0 0.1" range="-35 35" stiffness="10" type="hinge"/>
#                    <geom fromto="-.02 -.07 0 -.02 .07 0" name="butt" size="0.09" type="capsule"/>
#                    <body name="right_thigh" pos="0 -0.1 -0.04">
#                        <joint armature="0.01" axis="1 0 0" damping="5" name="right_hip_x" pos="0 0 0" range="-25 5" stiffness="10" type="hinge"/>
#                        <joint armature="0.01" axis="0 0 1" damping="5" name="right_hip_z" pos="0 0 0" range="-60 35" stiffness="10" type="hinge"/>
#                        <joint armature="0.0080" axis="0 1 0" damping="5" name="right_hip_y" pos="0 0 0" range="-110 20" stiffness="20" type="hinge"/>
#                        <geom fromto="0 0 0 0 0.01 -.34" name="right_thigh1" size="0.06" type="capsule"/>
#                        <body name="right_shin" pos="0 0.01 -0.403">
#                            <joint armature="0.0060" axis="0 -1 0" name="right_knee" pos="0 0 .02" range="-160 -2" type="hinge"/>
#                            <geom fromto="0 0 0 0 0 -.3" name="right_shin1" size="0.049" type="capsule"/>
#                            <body name="right_foot" pos="0 0 -0.45">
#                                <geom name="right_foot" pos="0 0 0.1" size="0.075" type="sphere" user="0"/>
#                            </body>
#                        </body>
#                    </body>
#                    <body name="left_thigh" pos="0 0.1 -0.04">
#                        <joint armature="0.01" axis="-1 0 0" damping="5" name="left_hip_x" pos="0 0 0" range="-25 5" stiffness="10" type="hinge"/>
#                        <joint armature="0.01" axis="0 0 -1" damping="5" name="left_hip_z" pos="0 0 0" range="-60 35" stiffness="10" type="hinge"/>
#                        <joint armature="0.01" axis="0 1 0" damping="5" name="left_hip_y" pos="0 0 0" range="-110 20" stiffness="20" type="hinge"/>
#                        <geom fromto="0 0 0 0 -0.01 -.34" name="left_thigh1" size="0.06" type="capsule"/>
#                        <body name="left_shin" pos="0 -0.01 -0.403">
#                            <joint armature="0.0060" axis="0 -1 0" name="left_knee" pos="0 0 .02" range="-160 -2" stiffness="1" type="hinge"/>
#                            <geom fromto="0 0 0 0 0 -.3" name="left_shin1" size="0.049" type="capsule"/>
#                            <body name="left_foot" pos="0 0 -0.45">
#                                <geom name="left_foot" type="sphere" size="0.075" pos="0 0 0.1" user="0" />
#                            </body>
#                        </body>
#                    </body>
#                </body>
#            </body>
#            <body name="right_upper_arm" pos="0 -0.17 0.06">
#                <joint armature="0.0068" axis="2 1 1" name="right_shoulder1" pos="0 0 0" range="-85 60" stiffness="1" type="hinge"/>
#                <joint armature="0.0051" axis="0 -1 1" name="right_shoulder2" pos="0 0 0" range="-85 60" stiffness="1" type="hinge"/>
#                <geom fromto="0 0 0 .16 -.16 -.16" name="right_uarm1" size="0.04 0.16" type="capsule"/>
#                <body name="right_lower_arm" pos=".18 -.18 -.18">
#                    <joint armature="0.0028" axis="0 -1 1" name="right_elbow" pos="0 0 0" range="-90 50" stiffness="0" type="hinge"/>
#                    <geom fromto="0.01 0.01 0.01 .17 .17 .17" name="right_larm" size="0.031" type="capsule"/>
#                    <geom name="right_hand" pos=".18 .18 .18" size="0.04" type="sphere"/>
#                    <camera pos="0 0 0"/>
#                </body>
#            </body>
#            <body name="left_upper_arm" pos="0 0.17 0.06">
#                <joint armature="0.0068" axis="2 -1 1" name="left_shoulder1" pos="0 0 0" range="-60 85" stiffness="1" type="hinge"/>
#                <joint armature="0.0051" axis="0 1 1" name="left_shoulder2" pos="0 0 0" range="-60 85" stiffness="1" type="hinge"/>
#                <geom fromto="0 0 0 .16 .16 -.16" name="left_uarm1" size="0.04 0.16" type="capsule"/>
#                <body name="left_lower_arm" pos=".18 .18 -.18">
#                    <joint armature="0.0028" axis="0 -1 -1" name="left_elbow" pos="0 0 0" range="-90 50" stiffness="0" type="hinge"/>
#                    <geom fromto="0.01 -0.01 0.01 .17 -.17 .17" name="left_larm" size="0.031" type="capsule"/>
#                    <geom name="left_hand" pos=".18 -.18 .18" size="0.04" type="sphere"/>
#                </body>
#            </body>
#        </body>
#    </worldbody>
#    <tendon>
#        <fixed name="left_hipknee">
#            <joint coef="-1" joint="left_hip_y"/>
#            <joint coef="1" joint="left_knee"/>
#        </fixed>
#        <fixed name="right_hipknee">
#            <joint coef="-1" joint="right_hip_y"/>
#            <joint coef="1" joint="right_knee"/>
#        </fixed>
#    </tendon>
#
#    <actuator>
#        <motor gear="100" joint="abdomen_y" name="abdomen_y"/>
#        <motor gear="100" joint="abdomen_z" name="abdomen_z"/>
#        <motor gear="100" joint="abdomen_x" name="abdomen_x"/>
#        <motor gear="100" joint="right_hip_x" name="right_hip_x"/>
#        <motor gear="100" joint="right_hip_z" name="right_hip_z"/>
#        <motor gear="300" joint="right_hip_y" name="right_hip_y"/>
#        <motor gear="200" joint="right_knee" name="right_knee"/>
#        <motor gear="100" joint="left_hip_x" name="left_hip_x"/>
#        <motor gear="100" joint="left_hip_z" name="left_hip_z"/>
#        <motor gear="300" joint="left_hip_y" name="left_hip_y"/>
#        <motor gear="200" joint="left_knee" name="left_knee"/>
#        <motor gear="25" joint="right_shoulder1" name="right_shoulder1"/>
#        <motor gear="25" joint="right_shoulder2" name="right_shoulder2"/>
#        <motor gear="25" joint="right_elbow" name="right_elbow"/>
#        <motor gear="25" joint="left_shoulder1" name="left_shoulder1"/>
#        <motor gear="25" joint="left_shoulder2" name="left_shoulder2"/>
#        <motor gear="25" joint="left_elbow" name="left_elbow"/>
#    </actuator>
#</mujoco>
#"""
#		self._mj_model = mujoco.MjModel.from_xml_string(xml_string)
#		self._mj_data  = mujoco.MjData(self._mj_model)
#		mujoco.mj_forward(self._mj_model, self._mj_data)
		
		print(xml_string)
		glfw.init()
		with mujoco.viewer.launch_passive(self._mj_model, self._mj_data, show_left_ui=True, show_right_ui=True) as viewer:
			self._viewer = viewer
			## TIME
			#start_time = time.time()
			# DEFAULT CAMERA POSITIONING
			viewer.cam.distance = self.size * 4
			viewer.cam.lookat = self.center
			desc = self.descendants['cameras']['descendants']
			while viewer.is_running():
				mujoco.mj_step(self._mj_model, self._mj_data)
				viewer.sync()
				# CALLBACK
				callback()
				#print(viewer.update_hfield.)
				#print(*sorted(viewer.__dir__()), sep='\n')
				# TIME SYNCH
				#model_time = self._mj_data.time
				#run_time   = time.time() - start_time
				#delta      = max(0, model_time - run_time)
				#time.sleep(delta)
			self._viewer = None


	@blue.restrict
	def step(self, 
		 n_steps: int|None       = 1, 
		 seconds: int|float|None = None, 
		 pbar:    bool           = True) -> None:
		"""
		Performs timesteps in the simmulation. Exactly one argument must be given by the user. 
		All cameras that are recording are taking images during the simulation at their respective 
		times.

		Parameters
		----------
		n_steps : int | None
			The number of steps to be taken
		seconds : int | float | None
			The number of seconds to be simulated
		"""
		#if not hasattr(self, '_mj_model') and not hasattr(self, '_mj_data'):
		if not self._built:
			raise Exception('The World must first be build before simulation steps can be performed. Use World.build()!')
		if n_steps is None and seconds is None:
			raise ValueError('Precisely one argument seconds or n_steps must be set, but both were None.')
		if seconds is not None:
			n_steps = int(seconds/self.timestep)
			if n_steps == 0:
				raise ValueError(f'With World.timestep={self.timestep} and seconds={seconds} the number of simulation steps is 0. Please decrease the timestep or increase the duration (seconds).')
		if pbar:
			iterator = tqdm.tqdm(range(n_steps))
		else:
			iterator = range(n_steps)
		for n in iterator:
			
			mujoco.mj_step(self._mj_model, self._mj_data)
			time = self._mj_data.time
			for camera, rec_dict in self._recordings.items():
				fps, last, writer = rec_dict['fps'], rec_dict['last'], rec_dict['writer']
				if last + 1/fps <= time:
					writer.append_data(camera.observation)
					rec_dict['last'] = time
		#for descendants in self.descendants.values():
		#	for descendant in descendants['descendants']:
		#		descendant._clear_step_cache()


	@blue.restrict
	def start_recording(self, 
			    *cameras: list[blue.CameraType], 
			    filename: str, 
			    fps:      int = 60,  
			    **kwargs):
		"""
		This method starts recordings of the cameras passed to it. The rendering is done 
		by the mujoco.Renderer using OpenGL. The user should make sure, that their OpenGL 
		instance is properly functioning. The video files are written to the ``recordings`` 
		directory next to the file instantiating the World. The format is infered from the 
		file extension in the ``filename`` using imageio. To set additional parameters for 
		the video formatting in imageio the user can set the ``**kwargs`` which are passed 
		to the imageio writer.

		Example
		-------
		If the user would like to record a gif image that loops forever (instead of 
		the default case of a single loop) the user can set the argument ``loop`` to 
		zero, indicating an infinitely looping gif in the imageio writer:

		>>> world.start_recording(camera, filename='my.gif', loop=0)

		Now the camera has started recording, but there is nothing yet happening in the 
		simulation. Using :meth:`World.step` the user can simulate the passage of time with 
		the recording cameras writing images to their respective video files at correpsonding 
		moments (as determined by ``fps`` and :attr:`World.timestep`). Additionally the user 
		can apply actions in between timesteps via an :class:`Agent <blueprints.agent.Agent>` 
		or through :class:`Actuators <blueprints.actuators.BaseActuator>` directly.

		For example, the following lines demonstrate such a simulation loop with some number 
		of steps in MDP space determined by ``time_steps``, some ``skip_frames``, a blueprints 
		:class:`Agent <blueprints.agent.Agent>` ``agent`` used to interface with the :class:`Sensors <blueprints.sensors.BaseSensor>` 
		and :class:`Actuators <blueprints.actuators.BaseActuator>` and an RL-Agent ``policy``
		
		>>> for t in range(time_steps):
		>>> 	    world.step(n_steps=skip_frames)
		>>> 	    agent.force = policy(agent.observations)

		Finally the recordings are stopped using :meth:`stop_recording` to save the recorded 
		videos to a file.

		>>> world.stop_recording()

		The final video of the users agent can then be found in ``'recordings/camera_my.gif'``

		Parameters
		----------
		*cameras : list [ blue.CameraType ]
			A list of :class:`Cameras <blueprints.camera.Camera>`
		filename : str
			The name (with a proper file extension indicating the format) of the video.
			If multiple cameras are passed, multiple videos will be created with the name 
			of each camera prefixed to the filename.
		fps : int
			The frames per second
		**kwargs : dict
			Additional keyword arguments that are passed to imageio to write the video.
		"""
		# VALUE CHECKS
		if not all(camera in self.all.cameras for camera in cameras):
			raise ValueError(f'Not all cameras given are attached within the kinematic hierarchy of the World.')
		if not self._built:
			raise Exception('World must first be build before recordings can start.')
		# MAKE DIRECTORIES
		path = f'{self._path}/recordings'
		if not os.path.exists(path):
			os.mkdir(path)
		for camera in cameras:
			writer = imageio.get_writer(f'{path}/{camera.name}_{filename}', fps=fps, **kwargs)
			self._recordings[camera] = {'fps': fps, 'writer': writer, 'last': 0.}


	@blue.restrict
	def stop_recording(self, 
			   *cameras: list[blue.CameraType]):
		"""
		This method stops recording for all the cameras passed. If no camera is passed, all recording cameras will be stopped.

		Parameters
		----------
		cameras : list [ blue.CameraType ]
			The cameras to be stopped.
		"""
		# VALUE CHECKS
		if not all(camera in self.all.cameras for camera in cameras):
			raise ValueError(f'Not all cameras given are attached within the kinematic hierarchy of the World.')
		if not cameras:
			for rec_dict in self._recordings.values():
				rec_dict['writer'].close()
			self._recordings = dict()
		else:
			for camera in cameras:
				self._recordings[camera]['writer'].close()
				del self._recordings[camera]


	@blue.restrict
	def to_xml_string(self) -> str:
		"""
		If not build, this method will first build the world, retrieve the XML string and then unbuild the world again. 
		If the user calls this method repeatedly the world should probably be build first.

		Returns
		-------
		str
			The xml string representing the xml model used to launch Mujoco.
		"""
		built = self._built
		if not built:
			self.build()
		string = xml.tostring(self._xml_root, encoding='unicode')
		if not built:
			self.unbuild()
		return string


	@blue.restrict
	@classmethod
	@blue._experimental
	def from_xml_string(cls, string: str) -> blue.WorldType:
		"""
		This method reconstructs a World from a Mujoco xml string. It currently only supports reading 
		models, that exclusively consist of blueprints supported features. In a future version it will 
		be extended to be compatible to all Mujoco models.
		
		Parameters
		----------
		string : str
			The xml description of the Mujoco model.
		
		Returns
		-------
		blue.WorldType
			The reconstructed World.
		"""
		init_args     = dict()
		caches        = defaultdict(dict)
		assets        = defaultdict(dict)
		sensors       = defaultdict(lambda: defaultdict(list))
		actuators     = defaultdict(lambda: defaultdict(list))
		ref_actuators = defaultdict(lambda: defaultdict(list))
		xml_tree      = xml.fromstring(string)
		xml_compiler  = xml_tree.find('compiler')
		xml_assets    = xml_tree.find('asset')
		xml_sensors   = xml_tree.find('sensor')
		xml_actuators = xml_tree.find('actuator')
		xml_worldbody = xml_tree.find('worldbody')
		xml_option    = xml_tree.find('option')
		# COLLECTING ARGS
		name = xml_tree.get('model')
		if name is not None:
			init_args['name'] = name
		if xml_compiler is not None:
			angle = xml_compiler.get('angle')
			if angle is not None:
				init_args['angle'] = angle
		if xml_option is not None:
			init_args['viscosity']  = float(xml_option.get('viscosity')) if xml_option.get('viscosity') is not None else 0.
			init_args['integrator'] = xml_option.get('integrator')
			xml_flag = xml_option.find('flag')
			if xml_flag is not None:
				init_args['gravity'] = xml_flag.get('gravity') == 'enable'
				init_args['contact'] = xml_flag.get('contact') == 'enable'
		if xml_sensors is not None:
			for sensor in xml_sensors:
				sensor_type   = sensor.tag
				sensor_obj    = blue.REGISTER._get_thing(sensor)
				sensor_parent = sensor_obj._PARENT_TYPE
				sensors[sensor_parent][sensor.get(sensor_parent)].append(sensor_obj)
		if xml_assets is not None:
			for asset in xml_assets:
				asset_type = asset.tag
				if asset_type in blue.REGISTER.CACHE_THINGS:
					asset_name = asset.get('name')
					if asset_name in caches[asset_type]:
						cache = caches[asset_type][asset_name]
					else:
						cache_class = blue.REGISTER.CACHE_THINGS[asset_type]
						cache = cache_class._from_xml_element(asset)
					# BUILD ASSET
					asset_class = blue.REGISTER._get_thing_class(asset)
					asset_obj   = asset_class._from_xml_element(xml_element=asset, 
											cache=cache)
				else:
					asset_obj = blue.REGISTER._get_thing(asset)
				assets[asset_type][asset.get('name')] = asset_obj
		if xml_actuators is not None:
			for actuator in xml_actuators:
				actuator_type = actuator.tag
				actuator_name = actuator.get('name') 
				actuator_obj  = blue.REGISTER._get_thing(actuator)
				actuator_obj._IGNORE_CHECKS = True
				if actuator_name in sensors['actuator']:
					actuator_obj.attach(*sensors['actuator'][actuator_name], copy=False)
				parent_name = actuator.get(actuator_obj._PARENT_REFERENCE)
				actuators[actuator_parent][parent_name].append(actuator_obj)
				if parent_name is not None:
						break
				#for actuator_parent in actuator_obj._PARENT_REFERENCE:
				#	parent_name = actuator.get(actuator_parent)
				#	if parent_name is not None:
				#		break
				for actuator_reference, reference_type in actuator_obj._OTHER_REFERENCES.items():
					reference_name = actuator.get(actuator_reference)
					if reference_name is not None:
						ref_actuators[reference_type][reference_name].append(actuator_obj)
		# INIT WORLD
		world = object.__new__(cls)
		world.__init__(**init_args)
		for xml_element in xml_worldbody:
			cls._build_from_xml(parent=world, 
						xml_element=xml_element, 
						assets=assets, 
						sensors=sensors, 
						actuators=actuators, 
						ref_actuators=ref_actuators)
		for tag_actuators in actuators.values():
			for name_parent in tag_actuators.values():
				for actuator in name_parent:
					actuator._IGNORE_CHECKS = False
		return world


	@property
	def model(self):
		"""
		If a feature is not accessable through blueprints you can 
		access the Mujoco model directly.

		Returns
		-------
		mj.Model
		"""
		if self._built:
			return self._mj_model
		else:
			raise Exception('Mujoco model is only accessable after the World has been built.')

	@property
	def data(self):
		"""
		If a feature is not accessable through blueprints you can 
		access the Mujoco data directly.

		Returns
		-------
		mj.Data
		"""
		if self._built:
			return self._mj_data
		else:
			raise Exception('Mujoco data is only accessable after the World has been built.')
	


	@blue.restrict
	@classmethod
	def _build_from_xml(cls,
			    parent, 
			    xml_element, 
			    assets:        dict, 
			    sensors:       dict, 
			    actuators:     dict, 
			    ref_actuators: dict) -> blue.ThingType:
		"""
		This method builds the world from the xml Element.
		
		Parameters
		----------
		parent : WorldType
			The World object to which the reconstructed Things are attached.
		xml_element : TYPE
			The xml element from which the Things used for reconstruction are taken.
		assets : dict
			A dictionary of assets.
		sensors : dict
			A dictionary of sensors.
		actuators : dict
			A dictionary of actuators.
		ref_actuators : dict
			A dictionary of ref_actuators.
		"""
		# CONSTRUCT ELEMENT
		xml_type  = xml_element.get('type')
		xml_tag   = xml_element.tag
		obj_class = blue.REGISTER._get_thing_class(xml_element)
		xml_args  = dict()
		if xml_type in assets:
			xml_name = xml_element.get(xml_type)
			xml_args['asset'] = assets[xml_type][xml_name]
		if xml_tag in sensors:
			xml_name = xml_element.get('name')
			xml_args['sensors'] = sensors[xml_tag][xml_name]
		if xml_tag in actuators:
			xml_name = xml_element.get('name')
			xml_args['actuators'] = actuators[xml_tag][xml_name]
		if xml_tag in ref_actuators:
			xml_name = xml_element.get('name')
			xml_args['ref_actuators'] = ref_actuators[xml_tag][xml_name]
		obj = obj_class._from_xml_element(xml_element=xml_element, **xml_args)
		# CONSTRUCT CHILDREN
		for xml_child in xml_element:
			cls._build_from_xml(parent=obj, 
						xml_element=xml_child, 
						assets=assets, 
						sensors=sensors, 
						actuators=actuators, 
						ref_actuators=ref_actuators)
		parent.attach(obj, copy=False)

	# PROPERTIES

	@property
	def size(self):
		"""
		The size is computed to sclae the FreeCamera of the viewer (see :meth:`World.view`).

		Returns
		-------
		np.ndarray
			The size of the World ignoring background Things.
		"""
		built = self._built
		if not built:
			self.build()
		size = self._size
		if not built:
			self.unbuild()
		return size


	@property
	def center(self):
		"""
		The center is computed to position the FreeCamera of the viewer (see :meth:`World.view`).

		Returns
		-------
		np.ndarray
			The center of the World ignoring background Things.
		"""
		built = self._built
		if not built:
			self.build()
		center = self._center
		if not built:
			self.unbuild()
		return center


	@property
	def texture(self) -> blue.SkyboxTextureType:
		"""
		Only a :class:`SkyboxTexture <blueprints.texture.Skybox>` are valid for the World.

		.. note::
			At the moment, Mujoco is only rendering SkyboxTextures of the World, if also another 
			texture is taken by a material refferenced by a Thing within the World.

		Returns
		-------
		blue.SkyboxTextureType
			The Skybox Texture which is applied to the background of the World.
		"""
		return self._texture


	@texture.setter
	@blue.restrict
	def texture(self, texture: blue.SkyboxTextureType|None) -> None:
		if texture is not None:
			self._texture = texture.copy()
			self._texture._parent = self
		else:
			self._texture = texture
