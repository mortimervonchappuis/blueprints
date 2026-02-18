"""
The Blueprints library interfaces the Mujoco (Multi Joints Contact) physics simulation. 
To construct a :class:`World <blueprints.world.World>` you can combine many :mod:`Things <blueprints.thing.base>` 
via :meth:`attachment <blueprints.thing.node.NodeThing.attach>`, while blueprints translates 
your construction to xml in the background. Things are differentiated in concrete Things, which 
are the onse that you can use directly and abstract classes which implement a utilities for multiple 
concrete Things. Abstract Things are mostly located in :mod:`thing <blueprints.thing>` and should not be instantiated directly. 
Never the less they contain documentation relevant for many concrete Things, so you might want to 
have a look at it. Concrete Things are:

Concrete Things
---------------
- :mod:`Actuators <blueprints.actuators>` (All but ``BaseActuator``)
- :mod:`Bodies <blueprints.body>`
- :mod:`Camera <blueprints.camera>`
- :mod:`Geoms <blueprints.geoms>` (All but ``BaseGeom``)
- :mod:`Sites <blueprints.sites>` (All but ``BaseSite``)
- :mod:`Geoms <blueprints.joints>` (All but ``BaseJoint``)
- :mod:`Light <blueprints.light>`
- :mod:`Materials <blueprints.material>`
- :mod:`Textures <blueprints.texture>` (All but ``BaseTexture``)
- :mod:`Sensors <blueprints.sensors>` (All but ``<Thing>Sensor``)
- :mod:`Tendons <blueprints.tendon>`  (Only ``Tendon``)
- :mod:`Worlds <blueprints.world>`

:mod:`Caches <blueprints.cache>` and :mod:`Assets <blueprints.assets>` are handeled in the background, so you don't 
have to create them manually and can most often treat their occurences in ordinary mujoco as 
another :meth:`attachment <blueprints.thing.node.NodeThing.attach>`.

Some helpful Things that are not part of the mujoco XML scheme but that are directly useable by 
the user are:

- :mod:`Agents <blueprints.agent>`
- :mod:`Lattices <blueprints.utils.lattice>`
- :mod:`Placeholders <blueprints.placeholder>`

Modificators
------------
Blueprints design is centered around the idea to 
reduce redundancy, that would be encountered in XML editing. Once you have created some Things 
you can easily produce on the fly modified copies through many helpful utility methods. We list 
some of the most important ones.


- :meth:`shift <blueprints.thing.moveable.MoveableThing.shift>`
- :meth:`locate <blueprints.thing.moveable.MoveableThing.locate>`
- :meth:`rotate <blueprints.thing.moveable.MoveableThing.rotate>`
- :meth:`align <blueprints.thing.moveable.MoveableThing.align>`
- :meth:`scaled <blueprints.thing.moveable.MoveableThing.scaled>`
- :meth:`Camera.looking <blueprints.camera.Camera.looking>`
- :meth:`copy <blueprints.thing.base.BaseThing.copy>`

The ``copy`` method in particular can be used to perform all sorts of modifications 
by passing the desired changes as keyword arguments.

Kinematic Tree
--------------
To modify the kinematic hierarchy there are some methods and attributes to have a look at:

- :attr:`parent <blueprints.thing.base.BaseThing.parent>` returns the parent of a Thing. If the Thing has no parent it returns ``None``.
- :attr:`root <blueprints.thing.base.BaseThing.root>` returns the root of a Things kinematic hierarchy (which might be itself).
- :attr:`path <blueprints.thing.base.BaseThing.path>` returns a list containing the path from a Thing to its root.
- :meth:`attach <blueprints.thing.node.NodeThing.attach>` attaches copies of any number of Things.
- :meth:`detach <blueprints.thing.node.NodeThing.detach>` reverses the effect of ``attach``.
- :meth:`all <blueprints.thing.node.NodeThing.all>` returns a :mod:`View <blueprints.utils.view>` of all children.
- :mod:`NodeThing.\\<children_type\\> <blueprints.utils.view>` returns a  :mod:`View <blueprints.utils.view>` of the given children type.

Alternatively kinematic trees can also be constructed by passing children as additional arguments to the ``__init__`` of 
all concrete :class:`NodeThings <blueprints.thing.node.NodeThing>`.

"""


from collections import defaultdict
from functools import wraps


WARNING_FLAG = defaultdict(lambda *args: False)


def _experimental(func):
	@wraps(func)
	def wrapper(*args, **kwargs):
		if not WARNING_FLAG[func.__name__]:
			WARNING_FLAG[func.__name__] = True
			print(f'WARNING: {func.__qualname__} is an experimental feature likely to cause problems!')
		return func(*args, **kwargs)
	return wrapper


from .utils.typechecker import restrict

from .types       import *
from .utils.view  import View, AllView, LatticeView

from .thing       import *

from .utils.lattice import Lattice
from .utils.perlin  import perlin
from .thing.focal import FocalThing

from .placeholder import Placeholder
from .body        import Body
from .tendon      import Tendon
from .world       import World

from . import assets
from . import tube
from . import sites
from . import geoms
from .utils import geometry
from . import sensors
from .utils import naming
from . import joints
from . import cache
from . import actuators

from .camera   import Camera
from .light    import Light
from .geoms    import BaseGeom
from .sites    import BaseSite
from .joints   import BaseJoint
from .material import Material
from .texture  import *
from .agent    import Agent

from .utils import register

#from .environment import Environment
from blueprints.utils.geometry import TAU, PI, DEGREES_TO_RADIANS, RADIANS_TO_DEGREES, Vector, Rotation

REGISTER = register.Register()