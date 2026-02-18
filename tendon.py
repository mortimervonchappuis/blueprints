"""
Tendons
=======

:class:`Tendons <blueprints.tendon.Tendon>` can be used to string together multiple Things. The can provide additional 
structural stability or connect otherwise kinetically distinct Things. In Mujoco, 
Tendon can be bound to Sites, Geoms and Joints according to certain constraints 
limiting which Things can be bound and which Tendon attributes are passed into the 
simulation. Tendons introduce cycles to the kinematic hierarchy and are therefore 
treated specially with the :meth:`Path.bind <blueprints.tendon.Path.bind>` method replacing :meth:`NodeThing.attach <blueprints.thing.node.NodeThing.attach>` within 
a special context manager.

Joint (Fixed) Tendons
---------------------
Joint Tendons are the simplest usecase since they only use a reduced set of 
attributes and do not support spliting paths into pulleys. The attributes allowed 
for Joint Tendons are:

* :attr:`name <blueprints.tendon.Tendon.name>`
* :attr:`limited <blueprints.tendon.Tendon.limited>`
* :attr:`range <blueprints.tendon.Tendon.range>`
* :attr:`frictionloss <blueprints.tendon.Tendon.frictionloss>`
* :attr:`margin <blueprints.tendon.Tendon.margin>`
* :attr:`springlength <blueprints.tendon.Tendon.springlength>`
* :attr:`stiffness <blueprints.tendon.Tendon.stiffness>`
* :attr:`damping <blueprints.tendon.Tendon.damping>`

We first create some example Things.

.. code-block:: py
	:caption: Creating Example Things

	>>> world = blue.World()
	>>> body  = blue.Body(geoms=blue.geoms.Sphere(), # introduces mass to Body
	>>>		      joints=blue.joints.Hinge())
	>>> world.attach(body.shift(x=0, name='A'),
	>>> 		 body.shift(x=3, name='B'),
	>>> 		 body.shift(x=6, name='C'))
	>>> tendon = blue.Tendon()

To bind the Joints to the tendon we first grab them from the world and then bind 
them to the tendon path within the Tendons context. When we bind a Joint to a Tendon, 
a coefficient must be specified weighing the interaction strength between Tendon and 
Joint.

.. code-block:: py
	:caption: Tendon Context

	>>> jA = world.bodies['A'].joints[0]
	>>> jB = world.bodies['B'].joints[0]
	>>> jC = world.bodies['C'].joints[0]
	>>> with tendon as path:
	>>> 	 path.bind(jA, coef=1.)
	>>> 	 path.bind(jB, coef=2.)
	>>> 	 path.bind(jC, coef=1.)

.. code-block:: py
	:caption: Alternatively

	>>> with tendon as path:
	>>> 	 path.bind(jA, jB, jC, coef=[1.0, 2.0, 1.0])

The resulting XML is split into the normal kinematic tree and the Tendon referencing 
its bound Things via name attribute.

.. code-block:: mxml
	:caption: XML Structure Tendon
	
	<tendon>
		<fixed name="anonymous_tendon">
			<joint joint="anonymous_hinge_(0)" coef="1.0" />
			<joint joint="anonymous_hinge_(1)" coef="2.0" />
			<joint joint="anonymous_hinge_(2)" coef="1.0" />
		</fixed>
	</tendon>


.. note::
	Keep in mind that Fixed Joint Tendons do not support :attr:`width <blueprints.tendon.Tendon.width>` and are hence invisible.


.. code-block:: mxml
	:caption: XML Structure World

	<worldbody>
		<body name="A">
			<joint type="hinge" name="anonymous_hinge_(0)" axis="0.0 0.0 1.0" />
			<geom size="1.0" type="sphere" name="anonymous_sphere_(0)" />
		</body>
		<body pos="3.0 0.0 0.0" name="B">
			<joint type="hinge" name="anonymous_hinge_(1)" axis="0.0 0.0 1.0" />
			<geom size="1.0" type="sphere" name="anonymous_sphere_(1)"/>
		</body>
		<body pos="6.0 0.0 0.0" name="C">
			<joint type="hinge" name="anonymous_hinge_(2)" axis="0.0 0.0 1.0" />
			<geom size="1.0" type="sphere" name="anonymous_sphere_(2)"/>
		</body>
	</worldbody>


Site/Geom (Spatial) Tendons
---------------------------
Spatial Tendons are bind to :class:`Sites <blueprints.sites.BaseSite>` and :class:`Geoms <blueprints.geoms.BaseGeom>` and support path splitting. The following rules
must be satisfied for mujoco to accept the build.

* Every path must start and end with a :class:`Site <blueprints.sites.BaseSite>`.
* Every :class:`Geom <blueprints.geoms.BaseGeom>` must be sandwiched between two :class:`Sites <blueprints.sites.BaseSite>`.
* Every Tendon binding Sites/Geoms must not also bind Joints.

Lets build some example Things and bind them to a Tendon.


.. code-block:: py
	:caption: Creating Example Things

	>>> body_A = blue.Body(sites=blue.sites.Sphere(radius=0.5)).shift(x=-3, name='A')
	>>> body_B = blue.Body(geoms=blue.geoms.Sphere(radius=0.5),
	>>> 		       sites=blue.sites.Sphere(radius=0.5).shift(z=3), name='B')
	>>> body_C = blue.Body(sites=blue.sites.Sphere(radius=0.5)).shift(x=3, name='C')
	>>> world.attach(body_A, body_C, body_B)
	>>> tendon = blue.Tendon(width=0.1)

When binding a :class:`Geom <blueprints.geoms.BaseGeom>` to a Tendon, an optional ``side_site`` argument may be 
specified. If the Tendon runs through the Geom at the begining of the Mujoco simulation, 
it is snapped out on the side of the specified ``side_site``.

.. code-block:: py
	:caption: Binding Sites and Geoms

	>>> sA = world.bodies['A'].sites[0]
	>>> sB = world.bodies['B'].sites[0]
	>>> gB = world.bodies['B'].geoms[0]
	>>> sC = world.bodies['C'].sites[0]
	>>> with tendon as path:
	>>> 	 path.bind(sA)
	>>> 	 path.bind(gB, side_site=sB)
	>>> 	 path.bind(sC)

.. image:: /_static/tendon_side_site.png

The resulting XML structure are separated into the cyclical references in the header of the Mujoco XML 
and its components in the normal kinematic hierarchy.

.. code-block:: mxml
	:caption: XML Structure Tendon

	<tendon>
		<spatial width="0.1" name="anonymous_tendon">
			<site site="anonymous_sphere_(0)" />
			<geom geom="anonymous_sphere" sidesite="anonymous_sphere_(2)" />
			<site site="anonymous_sphere_(1)" />
		</spatial>
	</tendon>

.. code-block:: mxml
	:caption: XML Structure World

	<worldbody>
		<body pos="-3.0 0.0 0.0" name="A">
		        <site size="0.5 0.0 0.0" type="sphere" name="anonymous_sphere_(0)" />
		</body>
		<body pos="3.0 0.0 0.0" name="C">
		        <site size="0.5 0.0 0.0" type="sphere" name="anonymous_sphere_(1)" />
		</body>
		<body name="B">
		        <geom size="0.5" type="sphere" name="anonymous_sphere" />
		        <site size="0.5 0.0 0.0" type="sphere" pos="0.0 0.0 3.0" name="anonymous_sphere_(2)" />
		</body>
	</worldbody>

Path Splitting
--------------
Mujoco supports pullies which split a path into multiple subpaths. The forces acting on each subpath 
are distributed proportionally. The begining of a new path is not necessarily directly connected to 
end of the parent path.

.. code-block:: py
	:caption: Creating Example Things

	>>> world   = blue.World()
	>>> body_A  = blue.Body(pos=[0, 0, 0], sites=blue.sites.Sphere(radius=0.5))
	>>> body_B  = blue.Body(pos=[2, 0, 0], sites=blue.sites.Sphere(radius=0.5))
	>>> body_C1 = blue.Body(pos=[4,-2, 0], sites=blue.sites.Sphere(radius=0.5))
	>>> body_C2 = blue.Body(pos=[4, 2, 0], sites=blue.sites.Sphere(radius=0.5))
	>>> body_D1 = blue.Body(pos=[6,-2, 0], sites=blue.sites.Sphere(radius=0.5))
	>>> body_D2 = blue.Body(pos=[6, 0, 0], sites=blue.sites.Sphere(radius=0.5))
	>>> body_D3 = blue.Body(pos=[6, 4, 0], sites=blue.sites.Sphere(radius=0.5))
	>>> tendon  = blue.Tendon(width=0.1)

Using the method :meth:`split <blueprints.tendon.Path.split>` a pulley is introduced that splits the path into `n` 
branches. A branch path can also be split.

.. code-block:: py
	:caption: Binding with Multiple Paths

	>>> with tendon as path:
	>>> 	 path.bind(body_A.sites)
	>>> 	 path.bind(body_B.sites)
	>>> 	 path_1, path_2, path_3 = path.split(3)
	>>> 	 # PATH ONE
	>>> 	 path_1.bind(body_B.sites)
	>>> 	 path_1.bind(body_C1.sites)
	>>> 	 path_1.bind(body_D1.sites)
	>>> 	 # PATH TWO
	>>> 	 path_2.bind(body_C2.sites)
	>>> 	 path_2.bind(body_D2.sites)
	>>> 	 # PATH THREE
	>>> 	 path_3.bind(body_C2.sites)
	>>> 	 path_3.bind(body_D3.sites)

The example above introduces a pulley with the ``divisor`` 3. The path bindings can be seen in the 
following image from left to right with ``path_1`` on the bottom and ``path_2`` and ``path_3`` ontop both 
sharing the same initial Site.

.. image:: /_static/tendon_example.png

The resulting XML structures are:

.. code-block:: mxml
	:caption: XML Structure Tendon

	<tendon>
                <spatial width="0.1" name="anonymous_tendon">
                        <site site="anonymous_sphere_(0)" />
                        <site site="anonymous_sphere_(1)" />
                        <pulley divisor="3" />
                        <site site="anonymous_sphere_(3)" />
                        <site site="anonymous_sphere_(6)" />
                        <pulley divisor="3" />
                        <site site="anonymous_sphere_(3)" />
                        <site site="anonymous_sphere_(5)" />
                        <pulley divisor="3" />
                        <site site="anonymous_sphere_(1)" />
                        <site site="anonymous_sphere_(2)" />
                        <site site="anonymous_sphere_(4)" />
                </spatial>
        </tendon>


.. code-block:: mxml
	:caption: XML Structure World

	<worldbody>
                <body name="anonymous_body_(0)">
                        <site size="0.5 0.0 0.0" type="sphere" name="anonymous_sphere_(0)" />
                </body>
                <body pos="2.0 0.0 0.0" name="anonymous_body_(1)">
                        <site size="0.5 0.0 0.0" type="sphere" name="anonymous_sphere_(1)" />
                </body>
                <body pos="4.0 -2.0 0.0" name="anonymous_body_(2)">
                        <site size="0.5 0.0 0.0" type="sphere" name="anonymous_sphere_(2)" />
                </body>
                <body pos="4.0 2.0 0.0" name="anonymous_body_(3)">
                        <site size="0.5 0.0 0.0" type="sphere" name="anonymous_sphere_(3)" />
                </body>
                <body pos="6.0 -2.0 0.0" name="anonymous_body_(4)">
                        <site size="0.5 0.0 0.0" type="sphere" name="anonymous_sphere_(4)" />
                </body>
                <body pos="6.0 0.0 0.0" name="anonymous_body_(5)">
                        <site size="0.5 0.0 0.0" type="sphere" name="anonymous_sphere_(5)" />
                </body>
                <body pos="6.0 4.0 0.0" name="anonymous_body_(6)">
                        <site size="0.5 0.0 0.0" type="sphere" name="anonymous_sphere_(6)" />
                </body>
        </worldbody>

Copies of Tendons
-----------------

Copying Tendons takes special attention since Tendons introduce cyclical references (i.e. multiple children of a :class:`NodeThing <blueprints.node.NodeThing>`
can be bound to the same Tendon). Therefore Tendons only get copied when the (root) Thing copied by the user has as its descendants all references 
of the Tendon.

Let's see an example of this:

.. code-block:: py
	:caption: Creating and Binding Example Things

	>>> tendon = blue.Tendon(width=0.1)
	>>> root = blue.Body(name='root')
	>>> body_site = blue.Body(sites=blue.sites.Sphere(radius=0.5))
	>>> root.attach(body_site.shift(x=0, name='A'), 
	>>> 		body_site.shift(x=2, name='B'))
	>>> A = root.bodies['A']
	>>> B = root.bodies['B']
	>>> C = body_site.shift(x=4, name='C')
	>>> with tendon as path:
	>>> 	 path.bind(A.sites)
	>>> 	 path.bind(B.sites)
	>>> 	 path.bind(C.sites)

We have created three bodies with sites (``A``, ``B``, ``C``) that are bound together with a Tendon. 
But only ``A`` and ``B`` are attached to root, while ``C`` is not. If both are attached to the World 
without creating a copy, they world will contain the tendon. 

.. code-block:: py
	:caption: World Attachment without Copy

	>>> world.attach(root, C, copy=False)

.. code-block:: mxml
	:caption: XML Structure Tendon

        <tendon>
                <spatial width="0.1" name="anonymous_tendon">
                        <site site="anonymous_sphere_(0)" />
                        <site site="anonymous_sphere_(1)" />
                        <site site="anonymous_sphere_(2)" />
                </spatial>
        </tendon>

.. code-block:: mxml
	:caption: XML Structure World

	
	<worldbody>
		<body name="root">
			<body name="A">
				<site size="0.5 0.0 0.0" type="sphere" name="anonymous_sphere_(0)" />
			</body>
			<body pos="2.0 0.0 0.0" name="B">
				<site size="0.5 0.0 0.0" type="sphere" name="anonymous_sphere_(1)" />
			</body>
		</body>
		<body pos="4.0 0.0 0.0" name="C">
			<site size="0.5 0.0 0.0" type="sphere" name="anonymous_sphere_(2)" />
		</body>
	</worldbody>


If we however create a copy of ``root`` and ``C`` separately the Tendon will be discarded, since neither 
``root`` nor ``C`` contain all refferences of the Tendon as their descendants (the references of the Tendon 
are the bound Sites ``"anonymous_sphere_(0)"``, ``"anonymous_sphere_(1)"`` and ``"anonymous_sphere_(2)"``).


.. code-block:: py
	:caption: World Attachment without Copy

	>>> world.attach(root, C, copy=True)

.. code-block:: mxml
	:caption: XML Structure Tendon

         </tendon>

.. code-block:: mxml
	:caption: XML Structure World

	<worldbody>
		<body name="root">
			<body name="A">
				<site size="0.5 0.0 0.0" type="sphere" name="anonymous_sphere_(0)" />
			</body>
			<body pos="2.0 0.0 0.0" name="B">
				<site size="0.5 0.0 0.0" type="sphere" name="anonymous_sphere_(1)" />
			</body>
		</body>
		<body pos="4.0 0.0 0.0" name="C">
			<site size="0.5 0.0 0.0" type="sphere" name="anonymous_sphere_(2)" />
		</body>
	</worldbody>
"""



from collections import defaultdict
import numpy as np
import xml.etree.ElementTree as xml
import blueprints as blue



class Tendon(blue.TendonType, blue.NodeThing, blue.ColoredThing, blue.FocalThing):
	@blue.restrict
	def __init__(self, 
		     name:	        str|None    = None, 
		     limited:	        bool|None   = None, 
		     act_force_limited: bool|None   = None, 
		     min_length:	int|float   = 0., 
		     max_length:	int|float   = 0., 
		     min_act_force:     int|float   = 0., 
		     max_act_force:     int|float   = 0., 
		     frictionloss:      int|float   = 0., 
		     width:	        int|float   = 0.003, 
		     color:	        object|None = None, 
		     stiffness:	        int|float   = 0., 
		     damping:	        int|float   = 0., 
		     armature:	        int|float   = 0.,
		     **kwargs) -> None:
		"""
		Some attribute descriptions are taken from `Mujoco <https://mujoco.readthedocs.io/en/latest/XMLreference.html#tendon>`__.

		Parameters
		----------
		limited : bool | None
			Sets whether length limits constraints are imposed on the mujoco 
			solver. Leaving this argument as None (recommended) enables autolimits 
			derived from the other attributes.
		act_force_limited : bool | None
			This argument is the equivalent to `limited` for actuator forces.
		min_length : int | float
			Minimal length of Tendon
		max_length : int | float
			Maximal length of Tendon
		min_act_force : int | float
			Minimal force allowed to act on Tendon
		max_act_force : int | float
			Maximal force allowed to act on Tendon
		frictionloss : int | float
			Friction loss caused by dry friction. This value should be positive.
		width : int | float
			Crosssection radius of the Tendon
		stiffness : int | float
			If set to a positive value it acts as a Spring coefficient.
		damping : int | float
			If set to a positive value it acts as a Damping coefficient.
		armature : int | float
			Inertia associated with changes in tendon length
		"""
		self._built      = False
		self._ACTIVE	    = False
		# MIGRATION ATTRIBUTES
		self._ADDRESS_BOOK  = defaultdict(list)
		self._MIGRATED      = False
		self._MIGRATIONS    = defaultdict(int)
		# PATH ATTRIBUTES
		self._branches = []
		# CHILDREN
		pack = lambda x: x if isinstance(x, list) else [x]
		self._sites      = []
		self._geoms      = []
		self._side_sites = []
		self._joints     = []
		self._CHILDREN	 = {}
		self._PSEUDO_CHILDREN  = {'sites':       {'type':     blue.SiteType, 
						 	  'children': self._sites}, 
					  'geoms':       {'type':     blue.GeomType, 
							  'children': self._geoms}, 
					  'side_sites':  {'type':     blue.SiteType, 
							  'children': self._side_sites},
					  'joints':      {'type':     blue.JointType, 
							  'children': self._joints}}
		# MUJOCO ATTRIBUTES
		self.limited	   = limited
		self.act_force_limited = act_force_limited
		self.min_length	= min_length
		self.max_length	= max_length
		self.min_act_force     = min_act_force
		self.max_act_force     = max_act_force
		self.frictionloss      = frictionloss
		self.width	     = width
		self.stiffness	 = stiffness
		self.damping	   = damping
		self.armature	  = armature
		# SETTING COLOR BLUEPRINT COLOR DEFAULT FOR GEOMS
		color = color if color is not None else 'grey'
		super().__init__(name=name, 
				 color=color,
				 **kwargs)


	def __enter__(self):
		self._ACTIVE = True
		self._branches = [Path(self)]
		return self._branches[0]


	def __exit__(self, exc_type, error, trace):
		self._ACTIVE = False


	@property
	def _MIGRATION_DONE(self):
		return all(v == 0 for v in self._MIGRATIONS.values())


	def _get_path(self, address):
		path = self
		for i in address:
			path = path._branches[i]
		return path


	def _migrate(self, old, new):
		if not self._MIGRATED:
			self._start_migration()
		if self._VALID:
			addresses = self._ADDRESS_BOOK[old]
			for address, i_path, other in addresses:
				path = self._COPY._get_path(address)
				if isinstance(old, blue.SiteType):
					path._path[i_path][other] = new
					if other == 0:
						self._COPY._sites.append(new)
						new._tendons.append(self._COPY)
					else:
						self._COPY._side_sites.append(new)
						new._side_tendons.append(self._COPY)
				elif isinstance(old, blue.GeomType):
					path._path[i_path][other] = new
					self._COPY._geoms.append(new)
					new._tendons.append(self._COPY)
				elif isinstance(old, blue.JointType):
					path._path[i_path][0] = new
					path._path[i_path][1] = other
					self._COPY._joints.append(new)
					new._tendons.append(self._COPY)

		self._MIGRATIONS[old] -= 1
		if self._MIGRATION_DONE:
			self._finalize_migration()


	def _start_migration(self):
		# CHECK LEGALITY
		self._MIGRATED = True
		self._VALID    = True
		things  = self._sites + self._geoms + self._side_sites + self._joints
		INVALID = []
		for thing in things:
			if blue.REGISTER.copy_root in thing.path:
				self._MIGRATIONS[thing] += 1
			else:
				self._VALID = False
				INVALID.append(thing)
		# CREATE COPY
		self._COPY = super().copy()
		# CREATE MIGRATION OBJECTS
		self._ADDRESS_BOOK = defaultdict(list)
		# CHECK VALIDITY
		if not self._VALID:
			print(f'WARNING: {repr(self)} is discarded. The Things ({', '.join(map(repr, INVALID))}) bound by the Tendon are not descendants of the copied Thing {repr(blue.REGISTER.copy_root)}.')
			return
		# BREADTH SEARCH
		queue = [([i], branch) for i, branch in enumerate(self._branches)]
		while queue:
			address, path = queue.pop(0)
			path_parent = self._COPY._get_path(address[:-1])
			path_copy = Path(self._COPY)
			path_parent._branches.append(path_copy)
			for i_path, (thing, other) in enumerate(path):
				self._ADDRESS_BOOK[thing].append((address, i_path, 0))
				if other is not None and isinstance(other, blue.SiteType):
					self._ADDRESS_BOOK[other].append((address, i_path, 1))
				path_copy._path.append([None, None])
			for i_branch, branch in enumerate(path._branches):
				queue.append((address + [i_branch], branch))
	

	def _finalize_migration(self):
		self._MIGRATED     = False
		self._VALID	= True
		self._MIGRATIONS   = defaultdict(int)
		del self._COPY
		del self._ADDRESS_BOOK


	def attach(self, *items, copy=False) -> None:
		"""
		Attachements are not allowed for Tendons. Use :meth:`Path.bind` instead.
		"""
		raise Exception('Attachement is not allowed for Tendons, use Path.bind instead. Refer to the Documentation for details.')


	def _build(self, 
		   parent, 
		   world, 
		   indicies, 
		   **kwargs):
		if self._built:
			return
		mujoco_specs = self._mujoco_specs(**kwargs)
		if self._joints:
			INVALID = []
			for attr in mujoco_specs.keys():
				if attr not in self._FIXED_ATTR:
					INVALID.append(attr)
			for attr in INVALID:
				del mujoco_specs[attr]
			if INVALID:
				print(f'WARNING: Tendons with Joints have a reduced set of attributes. Ignoring attributes ({', '.join(map(str, INVALID))}).')
		tendon = xml.SubElement(world._xml_tendon, 
					'fixed' if self._joints else 'spatial', 
					**mujoco_specs)
		queue = [(len(self._branches), path) for path in self._branches]
		while queue:
			idx, path = queue.pop()
			if idx > 1:
				pulley = xml.SubElement(tendon, 'pulley', divisor=str(idx))
			for thing, other in path:
				if isinstance(thing, blue.SiteType):
					xml.SubElement(tendon, 'site', site=thing.name)
				elif isinstance(thing, blue.GeomType):
					if not isinstance(other, blue.SiteType):
						xml.SubElement(tendon, 'geom', geom=thing.name)
					else:
						xml.SubElement(tendon, 'geom', geom=thing.name, sidesite=other.name)
				elif isinstance(thing, blue.JointType):
					if not isinstance(other, (int, float)):
						xml.SubElement(tendon, 'joint', joint=thing.name)
					else:
						xml.SubElement(tendon, 'joint', joint=thing.name, coef=str(float(other)))

			for branch in path._branches[::-1]:
				queue.insert(0, (len(path._branches), branch))
		self._built = True

	# DERIVED ATTRIBUTES

	@property
	def actuatorfrclimited(self) -> bool|None:
		"""
		Indicates whether the Actuator forces applied are limited.

		Returns
		-------
		bool | None
			In the attribute is None, the value is set to ``auto``.
		"""
		return self._act_force_limited


	@property
	def range(self) -> np.ndarray:
		"""
		The length range of the Tendon.

		Returns
		-------
		np.ndarray
			The two entries indicate minimum and maximum Tendon length.
		"""
		return np.array([self.min_length, 
				 self.max_length], dtype=np.float32)


	@property
	def actuatorfrcrange(self) -> np.ndarray:
		"""
		The Actuator force range of the Tendon.

		Returns
		-------
		np.ndarray
			The two entries indicate minimum and maximum Actuator force applied to the Tendon.
		"""
		return np.array([self.min_act_force, 
				 self.max_act_force], dtype=np.float32)

	# BLUEPRINTS ATTRIBUTES

	@property
	def max_length(self) -> float:
		"""
		Changing this value results in a change in the first component of ``range``.

		The user should make sure, that the initial length of the tendon as 
		computed by the distance between its bound Things does not exeed this attribute.

		Returns
		-------
		float
			Maximum length of the Tendon
		"""
		return self._max_length


	@max_length.setter
	@blue.restrict
	def max_length(self, max_length: int|float) -> None:
		self._max_length = float(max_length)


	@property
	def min_length(self) -> float:
		"""
		Changing this value results in a change in the second component of ``range``.
		
		Returns
		-------
		float
			Minimum length of the Tendon
		"""
		return self._min_length


	@min_length.setter
	@blue.restrict
	def min_length(self, min_length: int|float) -> None:
		self._min_length = float(min_length)


	@property
	def max_act_force(self) -> float:
		"""
		Changing this value results in a change in the first component of ``actuatorfrcrange``.

		Returns
		-------
		float
			Maximum force applicable by an :class:`Actuator <blueprints.actuator.BaseActuator>`
		"""
		return self._max_act_force


	@max_act_force.setter
	@blue.restrict
	def max_act_force(self, max_act_force: int|float) -> None:
		self._max_act_force = float(max_act_force)


	@property
	def min_act_force(self) -> float:
		"""
		Changing this value results in a change in the second component of ``actuatorfrcrange``.

		Returns
		-------
		float
			Minimum force applicable by an :class:`Actuator <blueprints.actuator.BaseActuator>`
		"""
		return self._min_act_force


	@min_act_force.setter
	@blue.restrict
	def min_act_force(self, min_act_force: int|float) -> None:
		self._min_act_force = float(min_act_force)

	# MUJOCO ATTRIBUTES

	@property
	def limited(self) -> bool|None:
		"""
		Indicates whether the Tendon length is limited by ``range``.

		Returns
		-------
		bool | None
			In the attribute is None, the value is set to ``auto``.
		"""
		return self._limited


	@limited.setter
	@blue.restrict
	def limited(self, limited: bool|None) -> None:
		self._limited = limited


	@property
	def act_force_limited(self) -> bool|None:
		"""
		Indicates whether the Actuator forces applied are limited.

		Returns
		-------
		bool | None
			In the attribute is None, the value is set to ``auto``.
		"""
		return self._act_force_limited


	@act_force_limited.setter
	@blue.restrict
	def act_force_limited(self, act_force_limited: bool|None) -> None:
		self._act_force_limited = act_force_limited


	@property
	def frictionloss(self) -> float:
		"""
		Friction loss caused by dry friction.

		Returns
		-------
		float
			To enable friction loss, set this attribute to a positive value.
		"""
		return self._frictionloss


	@frictionloss.setter
	@blue.restrict
	def frictionloss(self, frictionloss: int|float) -> None:
		self._frictionloss = float(frictionloss)


	@property
	def width(self) -> float:
		"""
		This attribute is purely cosmetical.

		Returns
		-------
		föoat
			Width of the Tendon.
		"""
		return self._width


	@width.setter
	@blue.restrict
	def width(self, width: int|float) -> None:
		self._width = float(width)


	@property
	def stiffness(self) -> float:
		"""
		A positive value generates a spring force (linear in position) acting along the Tendon.

		Returns
		-------
		float
			Stiffness of the Tendon
		"""
		return self._stiffness


	@stiffness.setter
	@blue.restrict
	def stiffness(self, stiffness: int|float) -> None:
		self._stiffness = float(stiffness)


	@property
	def damping(self) -> float:
		"""
		A positive value generates a damping force (linear in velocity) acting along the tendon. 
		If possible, :attr:`Joint.damping <blueprints.joints.BaseJoint.damping>` should be used 
		for stability.

		Returns
		-------
		float
			Tendon dampin.
		"""
		return self._damping


	@damping.setter
	@blue.restrict
	def damping(self, damping: int|float) -> None:
		self._damping = float(damping)


	@property
	def armature(self) -> float:
		"""
		Setting this attribute to a positive value :math:`m` adds a kinetic 
		energy term :math:`\\frac{1}{2}mv^2`, where :math:`v` is the tendon velocity.

		Returns
		-------
		float
			Inertia associated with changes in Tendon length
		"""
		return self._armature


	@armature.setter
	@blue.restrict
	def armature(self, armature: int|float) -> None:
		self._armature = float(armature)



class Path(blue.PathType):
	"""
	Paths are used to specify the ``<pulley>`` element in Mujoco. 
	A path can be strung along multiple :class:`Sites <blueprints.sites.BaseSite>`, 
	:class:`Joints <blueprints.joints.BaseJoint>` and :class:`Geoms <blueprints.geoms.BaseGeom>`. 
	Every path must start and end with a Site and every Geom must be sandwitched between two Sites.
	"""
	def __init__(self, tendon):
		"""
		.. warning::
			This class is not to be instantiated by the user directly. Use the contextmanager 
			:meth:`Tendon.__enter__` instead (see above).
		"""
		self.tendon = tendon
		self._path  = []
		self._branches = []
		self._split = False


	def __iter__(self):
		for thing, side_site in self._path:
			yield thing, side_site


	def __getitem__(self, idx):
		if isinstance(idx, int):
			if idx >= len(self._path):
				raise IndexError(f'This Path has only {len(self._path)} entries but entry {idx} was requested!')
			return self._path[idx][0]
		elif isinstance(idx, slice):
			if idx.start >= len(self._path):
				raise IndexError(f'This Path has only {len(self._path)} entries but entry {idx.start} was requested!')
			elif idx.stop >= len(self._path):
				raise IndexError(f'This Path has only {len(self._path)} entries but entry {idx.stop} was requested!')
			return [thing for thing, side_site in self._path[idx]]
		else:
			raise NotImplemented


	@blue.restrict
	def bind(self, 
		 *things:   list[blue.ThingType], 
		 side_site: blue.SiteType|None = None,
		 coef:      int|float|list[int|float]|None = None) -> None:
		"""
		Binding is performed in order. If a ``side_site`` is specified only 
		one Geom to be bound can be passed.

		Parameters
		----------
		*things : list [ blue.ThingType ]
			The Sites and Geoms to be bound to a Tendon path
		side_site: blue.SiteType | None
			A side Site for a Geom specifies to which side of the geom 
			the Tendon should snap out if passing through the Geom initially.
		coef : int | float | list [ int | float] | None
			Numeric coefficients for Joints. The argument must have the same shape as *things.
		"""
		if self._split:
			raise Exception(f'Paths can not bind Things after they were split. ')
		if not self.tendon._ACTIVE:
			raise Exception('Tendons can only be bound to Things within the context manager. Use "with tendon as path:" and refer to the Documentation.')
		if side_site is not None:
			if len(things) != 1 or not isinstance(things[0], blue.GeomType):
				raise ValueError('If a side_site is specified, precisely one Geom must be given.')
		# FLATTENING VIEWS
		flattened_things = []
		for thing in things:
			if isinstance(thing, blue.ViewType):
				flattened_things.extend(list(thing))
			else:
				flattened_things.append(thing)
		# CHECKING COEFFICIENTS
		if coef is not None:
			if not all(isinstance(thing, blue.JointType) for thing in flattened_things):
				raise ValueError(f'Coefficients can only be specified for Joints.')
			coef = [coef] if not isinstance(coef, list) else coef
			if len(coef) != len(things):
				raise ValueError(f'Every Coefficient must correspond to a Joint. Got {len(coef)} Coefficients and {len(things)} Joints.')
		# BINDING THINGS
		for i, thing in enumerate(flattened_things):
			if isinstance(thing, blue.SiteType):
				if self.tendon._joints:
					raise ValueError('Tendons can either bind Joints or Sites and Geoms, not both!')
				self.tendon._sites.append(thing)
			elif isinstance(thing, blue.GeomType):
				if isinstance(self._path[-1], blue.GeomType):
					raise ValueError('Geoms bound to a Tendon must be sandwiched by Sites!')
				if self.tendon._joints:
					raise ValueError('Tendons can either bind Joints or Sites and Geoms, not both!')
				if not isinstance(thing, (blue.SphereGeomType, blue.CylinderGeomType)):
					raise ValueError(f'The only geom types allowed for Tendon binding are Cylinders and Spheres. Got {type(thing)}.')
				self.tendon._geoms.append(thing)
				if side_site is not None:
					self.tendon._side_sites.append(side_site)
					side_site._side_tendons.append(self.tendon)
			elif isinstance(thing, blue.JointType):
				if self.tendon._geoms or self.tendon._sites:
					raise ValueError('Tendons can either bind Joints or Sites and Geoms, not both!')
				self.tendon._joints.append(thing)
			else:
				raise ValueError(f'Tendon.bind expects Things of Type Geom, Site or Joint. Got {', '.join(map(lambda x: str(repr(x)), flattened_things))}')
			other = coef[i] if coef is not None and isinstance(thing, blue.JointType) else side_site
			self._path.append([thing, other])
			thing._tendons.append(self.tendon)


	def split(self, number: int) -> tuple:
		"""
		If a pulley should split the Tendon into :math:`n` :class:`Paths <blueprints.tendon.Path>`, this method 
		handles the split. A split Path does not necessarily start at the last element of its parent Path. 
		Once split, a Path can no longer bind Things.

		Parameters
		----------
		number : int
			The number of new Paths

		Returns
		-------
		tuple [ blue.PathType ]
			The new Paths connected through a pulley
		"""
		if not self.tendon._ACTIVE:
			raise Exception('Paths can only be split within the context manager. Use "with tendon as path:" and refer to the Documentation.')
		if self.tendon._joints:
			raise ValueError('Tendons binding Joints cannot be split into pulleys!')
		self._branches = list(Path(self.tendon) for _ in range(number))
		self._split = True
		return tuple(iter(self._branches))