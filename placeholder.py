"""
Placeholders can be attached to a :class:`blueprints.body.Body` to mark prominent locations. They do not appear 
in the resulting xml model and cannot be reconstructed from xml. Is is used to mimic a node in the kinematic 
tree that can be used to specify default attachment locations. If a Thing is attached to a Placeholder it is 
instead attached to the Placeholders parent in a position and orienation as if it would have been attached to 
the Placeholder directly. This can either be used for conveniance if many Things are attached to the same spot 
on the parent Body or in constructing a template with predefined spots for attachment. The following example 
shows how to build the template of a table.

.. code-block::
	:caption: Example table construction

	>>> top = blue.geoms.Box(name='top', x_length=2, y_length=1, z_length=0.002)
	>>> leg = blue.geoms.Box.from_points([0, 0, 0], [0, 0, 1], radius=0.03, name='leg')
	>>> table = blue.Body(name='table', pos=[0, 0, 1], geoms=top)
	>>> table.attach(blue.Placeholder(pos=[ 0.9, 0.4, 0], alpha=PI), 
	>>>              blue.Placeholder(pos=[ 0.9,-0.4, 0], alpha=PI), 
	>>>              blue.Placeholder(pos=[-0.9, 0.4, 0], alpha=PI), 
	>>>              blue.Placeholder(pos=[-0.9,-0.4, 0], alpha=PI))
	>>> table.placeholders.attach(leg)
	>>> table.geoms
	View[5|table.geoms]
	>>> table.attach(blue.Placeholder(name='plate', pos=[-0.7, 0, 0]), 
	>>>              blue.Placeholder(name='plate', pos=[ 0.7, 0, 0]))
	>>> table.placeholders
	['leg_(0)', 'leg_(1)', 'leg_(2)', 'leg_(3)', 'plate_(0)', 'plate_(1)']

.. code-block:: mxml
	:caption: XML structure

	<body pos="0.0 0.0 1.0" name="table">
		<geom size="1.0 0.5 0.001" type="box" name="top" />
		<geom size="0.015 0.015 0.5" type="box" euler="3.1415927 0.0 0.0" pos=" 0.9  0.4 -0.5" name="leg_(0)" />
		<geom size="0.015 0.015 0.5" type="box" euler="3.1415927 0.0 0.0" pos=" 0.9 -0.4 -0.5" name="leg_(1)" />
		<geom size="0.015 0.015 0.5" type="box" euler="3.1415927 0.0 0.0" pos="-0.9  0.4 -0.5" name="leg_(2)" />
		<geom size="0.015 0.015 0.5" type="box" euler="3.1415927 0.0 0.0" pos="-0.9 -0.4 -0.5" name="leg_(3)" />
	</body>
"""
import numpy as np
import blueprints as blue



class Placeholder(blue.PlaceholderType, blue.MoveableThing):

	"""
	This class is available through the shortcut :class:`blueprints.Placeholder <Placeholder>`.

	Placeholders can be attached to a :class:`blueprints.body.Body` to mark special places and orientations 
	for attachment. It will not be included into the xml.

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
	name : str | None, optional
		The user specified name of the Body. In the case of a naming conflict the name is appended by an enumeration scheme.
	**kwargs
		Keyword arguments are passed to ``super().__init__``.

	Attributes
	----------
	parent : blue.NodeThingType | None
		When a Thing is attached to the Placeholder, it instead gets attached to the parent and its position 
		and orientation will be adjusted such that it is positioned and oriented as if it were attached of 
		the Placeholder.
	"""
	
	def attach(self, *args):
		"""
		Things that are attached to a Placeholder are instead attached to the Placeholders parent Body 
		in a position and orientation as if they were attached to the Placeholder directly.
		
		Parameters
		----------
		*args
			the Things that sould be attached to the Placeholders :attr:`parent`
		
		Raises
		------
		Expection
			If the Placeholder is not attached to a parent an error is raised.
		"""
		if self.parent is None:
			raise Expection('Placeholders must be attached to a body before a Thing can be attached to them!')
		else:
			args = [arg.rotate(self.alpha, 
					   self.beta, 
					   self.gamma, 
					   center=np.zeros(3), 
					   globally=False).shift(self.pos) for arg in args]
			self.parent.attach(*args)


	def detach(self, *args):
		"""
		This method redirects to the Placeholders parent. This means, that Things that have not been 
		attached to the parent directly (not via the Placeholder) can also be detached with this method.
		
		Parameters
		----------
		*args
			the Things that sould be detached from the Placeholders :attr:`parent`
		"""
		if self.parent is None:
			raise Expection('Placeholders must be attached to a body before a Thing can be detached from them!')
		self.parent.detach(*args)
		


	@blue.restrict
	def _build(self, 
		   parent, 
		   world, 
		   indicies, 
		   **kwargs):
		"""
		Placeholders are not part of mujoco xml.
		
		Parameters
		----------
		parent : xml.etree.ElementTree.Element
			The parent of the Thing
		world : blue.WorldType
			The World from which the initial _meth:`blueprints.world.World.build` method was called.
		"""
		pass
