"""
Things are base classes used to implement the rudimentary features of mujoco objects. All 
objects that are used to build a kinematic tree in blueprints are Things. A variety of 
Things can be created and combined in a kinematic structure to design your simulation. 
Most of the time you will create objects that are used multiple times in your simulation.
Therefore, once a Thing is created it serves as a blueprint. Attaching it to another Thing 
will create a copy, such that the blueprint is unchanged and can be altered and attached at 
different nodes in the kinematic tree without changing the the previously attach version 
of itself.

:class:`base.BaseThing` is the class from which all objects in the kinematic tree ultimately derive. 
It handles the basic methods used to translate from blueprints in python to mujoco xml.

:class:`node.NodeThing` is used to implement methods that are used to create structure for the 
kinematic tree. Every Thing that can have children attached to it is a :class:`node.NodeThing`.

:class:`moveable.MoveableThing` implements position and rotation attributes as well as methods to 
create rotated and shifted copies of itself.

:class:`colored.ColoredThing` contains attributes for color and opacity.

:class:`unique.UniqueThing` are objects that should be copied seldomly. Those are Things to which 
multiple other Things might refer and that should not be duplicated arbitrarily. Some Things 
like :class:`blueprints.geoms.Mesh` will appear to the user as one object, but are actually 
separated into multiple objects in the background to store large datasets separately. This 
enables memory efficiency for copies of the mesh that do not alter the dataset contained.

:class:`cyclical.CyclicalThing` are used for all Things that might refer to multiple other Things in 
such that it creates a graph cycle.

"""

from .base     import *
from .node     import *
from .moveable  import *
from .colored  import *
from .unique   import *
from .cyclical import *
