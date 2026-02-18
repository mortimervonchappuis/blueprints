"""
Construction and Attachement
----------------------------

Lattices are a blueprints native Thing. They are not part of the mujoco xml but rather 
a convenience object to help generate copies of a :class:`MoveableThing <blueprints.thing.moveable.MoveableThing>` 
arranged in a grid. Attaching a Lattice to any :class:`NodeThing <blueprints.thing.node.NodeThing>` results 
in its components being attached instead. A Lattice can also be constructed from another Lattice


.. code-block:: python
	:caption: Lattice Attachment:

	>>> world   = blue.World()
	>>> box     = blue.geoms.Box(z=0.5, color='grey')
	>>> lattice = box.lattice(directions=[[2, 0, 0], [0, 1.5, 0]], repetitions=[4, 5])
	>>> world.attach(lattice)
	>>> world.attach(blue.geoms.Plane(color='white'), blue.Light(z=10))


.. code-block:: xml
	:caption: XML Structure:

	<worldbody>
                <light cutoff="360.0" name="anonymous_light" pos="0.0 0.0 10.0" />
                <geom size="0.5 0.5 0.5" type="box" name="anonymous_box_(0)" pos="0.0 0.0 0.5" />
                <geom size="0.5 0.5 0.5" type="box" name="anonymous_box_(1)" pos="2.0 0.0 0.5" />
                <geom size="0.5 0.5 0.5" type="box" name="anonymous_box_(2)" pos="4.0 0.0 0.5" />
                <geom size="0.5 0.5 0.5" type="box" name="anonymous_box_(3)" pos="6.0 0.0 0.5" />
                <geom size="0.5 0.5 0.5" type="box" name="anonymous_box_(4)" pos="0.0 1.5 0.5" />
                <geom size="0.5 0.5 0.5" type="box" name="anonymous_box_(5)" pos="2.0 1.5 0.5" />
                <geom size="0.5 0.5 0.5" type="box" name="anonymous_box_(6)" pos="4.0 1.5 0.5" />
                <geom size="0.5 0.5 0.5" type="box" name="anonymous_box_(7)" pos="6.0 1.5 0.5" />
                <geom size="0.5 0.5 0.5" type="box" name="anonymous_box_(8)" pos="0.0 3.0 0.5" />
                <geom size="0.5 0.5 0.5" type="box" name="anonymous_box_(9)" pos="2.0 3.0 0.5" />
                <geom size="0.5 0.5 0.5" type="box" name="anonymous_box_(10)" pos="4.0 3.0 0.5" />
                <geom size="0.5 0.5 0.5" type="box" name="anonymous_box_(11)" pos="6.0 3.0 0.5" />
                <geom size="0.5 0.5 0.5" type="box" name="anonymous_box_(12)" pos="0.0 4.5 0.5" />
                <geom size="0.5 0.5 0.5" type="box" name="anonymous_box_(13)" pos="2.0 4.5 0.5" />
                <geom size="0.5 0.5 0.5" type="box" name="anonymous_box_(14)" pos="4.0 4.5 0.5" />
                <geom size="0.5 0.5 0.5" type="box" name="anonymous_box_(15)" pos="6.0 4.5 0.5" />
                <geom size="0.5 0.5 0.5" type="box" name="anonymous_box_(16)" pos="0.0 6.0 0.5" />
                <geom size="0.5 0.5 0.5" type="box" name="anonymous_box_(17)" pos="2.0 6.0 0.5" />
                <geom size="0.5 0.5 0.5" type="box" name="anonymous_box_(18)" pos="4.0 6.0 0.5" />
                <geom size="0.5 0.5 0.5" type="box" name="anonymous_box_(19)" pos="6.0 6.0 0.5" />
                <geom size="0.0 0.0 1.0" type="plane" name="anonymous_plane" rgba="1.0 1.0 1.0 1.0" />
        </worldbody>


.. image:: /_static/lattice_example_1.png

This Lattice made from a Lattice of a Box is istelf ofcause just a Lattice of a Box constructed by joining 
the axis of the two Lattices: ``lattice_A`` :math:`\\cong` ``lattice_B``


.. code-block:: python
	:caption: Lattice Creation:

	>>> lattice_B = blue.Lattice(box, directions=[[ 2., 0.0, 0.], 
	>>> 					      [ 0., 1.5, 0.], 
	>>> 					      [10., 0.0, 0.], 
	>>> 					      [ 0., 9.0, 0.]], 
	>>> 				  repetitions=[4, 5, 6, 7])


.. image:: /_static/lattice_example_2.png


Indexing and Attributes
-----------------------
A Lattice can be indexed via tuples of integer and slices. If less axes are indexed than contained by 
the Lattice, the last axes are retrieved fully. Indexed Lattices utilize a :class:`LatticeView <blueprints.utils.view.LatticeView>` 
which handles attribute retrieval and setting.


.. code-block:: python
	:caption: One to One:

	>>> box = blue.geoms.Box(x_length=0.03, 
	>>> 			 y_length=0.01, 
	>>> 			 z_length=0.06, 
	>>> 			 color='grey')
	>>> domino = blue.Body(geoms=box, 
	>>> 		       joints=blue.joints.Free(), 
	>>> 		       z=0.02)
	>>> row = domino.lattice([0, 0.06, 0], [20])
	>>> row[0].alpha = -TAU/32
	>>> row[0].y = 0.0075


.. image:: /_static/domino.gif

In the above example, we start the domino chain by tipping over a single Body indexed by ``row[0]``. 
We can instead also modify multiple ellements with a single value, setting every second domino color to ``'black'`` and ``'white'``.

.. code-block:: python
	:caption: One to Many:

	>>> row[0::2].geoms.color = 'black'
	>>> row[1::2].geoms.color = 'white'

.. image:: /_static/black_white_domino.gif


Lastly we can also assign many values to many elements by providing a list which is mapped one to one to the Lattices components. 
We create a gradient of rainbow colors in the same shape as the Lattice.


.. code-block:: python
	:caption: Many to Many:

	>>> rainbow = blue.gradient('purple', 'red', 'orange', 'yellow', 'green', 'teal', 'blue', 'purple', n_steps=20) 
	>>> row.color = rainbow


.. image:: /_static/rainbow_domino.gif

"""



import numpy as np
import blueprints as blue



class Lattice(blue.LatticeType):
	"""
	A Lattice is constructed from a Thing, a set of directions or axes and the number of repetitions. 
	Lattices are not mujoco objects but blueprints utility.
	"""
	__RESERVED = {'name', 
		      '_directions', 
		      '_repetitions', 
		      '_thing', 
		      '_things'}
	@blue.restrict
	def __init__(self, 
		     thing:       blue.MoveableThingType|blue.LatticeType, 
		     directions:  list[float|int]|np.ndarray|list[list[int|float]]|list[np.ndarray], 
		     repetitions: int|list[int]|np.ndarray, 
		     name:        str|None = None) -> None:
		"""
		Parameters
		----------
		thing : blue.MoveableThingType | blue.LatticeType
			This Thing will consitute the content of the Lattice
		directions : list [ float | int ] | np.ndarray | list [ list [ int | float ] ] | list [ np.ndarray ] 
			A list of axes pointing into the directions of the Lattice.
		repetitions : int | list [ int ] | np.ndarray
			A list of repetitions for each axis with the same length as ``directions``
		name : str | None
			The name of the Lattice (used for displaying purposes).
		"""
		self.name = name if name is not None else 'anonymous_lattice'
		# FORMAT DIRECTIONS
		if isinstance(directions, np.ndarray):
			directions = [directions]
		elif all(isinstance(val, (int, float)) for val in directions):
			directions = [directions]
		if all(isinstance(direction, list) for direction in directions):
			directions = [np.array(direction, dtype=np.float32) for direction in directions]
		self._directions = directions
		# FORMAT REPETITIONS
		if isinstance(repetitions, np.ndarray):
			repetitions = list(map(int, repetitions))
		if isinstance(repetitions, int):
			repetitions = [repetitions]
		self._repetitions = repetitions
		if isinstance(thing, blue.LatticeType):
			# EXTEND AXIS
			self._directions  += thing._directions.copy()
			self._repetitions += thing._repetitions.copy()
			# FORMAT THINGS
			self._thing  = thing._thing.copy()
			self._things = self._create_things(self._thing)
		else:
			# FORMAT THINGS
			self._thing  = thing.copy()
			self._things = self._create_things(self._thing)


	@property
	def n_dim(self) -> int:
		"""
		The number of dimensions in the Lattice
		
		Returns
		-------
		int
		"""
		return len(self._directions)


	def lattice(self, 
		    directions:  list[float|int]|np.ndarray|list[list[int|float]]|list[np.ndarray], 
		    repetitions: int|list[int]|np.ndarray, 
		    name:        str|None = None) -> None:
		"""
		A Lattice created with the parent Lattice as its content Thing (see examples above).
		
		Returns
		-------
		blue.LatticeType
		"""
		
		return blue.Lattice(thing=self, 
				    directions=directions, 
				    repetitions=repetitions, 
				    name=name)


	def __repr__(self) -> str:
		return f'Lattice[{'×'.join(map(str, self._repetitions))}|{self._thing.name}]'


	def _create_things(self, thing):
		def rec(depth, field, direction):
			if depth == 1:
				return [thing.shift(direction) for thing in field]
			else:
				return [rec(depth - 1, subfield, direction) for subfield in field]
		for depth, direction, repetition in zip(range(self.n_dim), self._directions[::-1], self._repetitions[::-1]):
			if depth == 0:
				field = [thing.shift(i * direction) for i in range(repetition)]
			else:
				field = [rec(depth, field, i * direction) for i in range(repetition)]
		return field


	def __iter__(self):
		for item in blue.LatticeView(self._things, self):
			yield item


	def copy(self, **kwargs):
		"""
		Parameters
		----------
		kwargs : dict
			The keyword arguments are passed to the individual component Things copy method.
		
		Returns
		-------
		blue.LatticeType
			A fresh copy of the Lattice
		"""
		
		def rec(depth, field, kwargs):
			if depth == 1:
				return [thing.copy(**kwargs) for thing in field]
			else:
				return [rec(depth - 1, subfield, kwargs) for subfield in field]
		things  = rec(self.n_dim, self._things, kwargs)
		# MIRROR INIT
		lattice = object.__new__(Lattice)
		lattice.name         = self.name
		lattice._directions  = self._directions.copy()
		lattice._repetitions = self._repetitions.copy()
		lattice._thing       = self._thing.copy(**kwargs)
		lattice._things      = things
		return lattice


	def shift(self, 
		  pos: list[int|float]|np.ndarray|None                        = None, 
		  x:   int|float|np.int32|np.int64|np.float32|np.float64|None = None, 
		  y:   int|float|np.int32|np.int64|np.float32|np.float64|None = None, 
		  z:   int|float|np.int32|np.int64|np.float32|np.float64|None = None, 
		  **kwargs):
		"""
		This method creates a copy that is shifted by ``pos``.
		
		Parameters
		----------
		pos : list[int | float] | np.ndarray | blue.MoveableThingType | None
			The amount by which the Thing is shifted, optionally another :class:`MoveableThing` 
			can be passed in which case the position of this Thing is taken.
		globally : bool, optional
			Determining whether the shift is relative to the local orientation or the global orientation.
		x : int | float | np.int32 | np.int64 | np.float32 | np.float64 | None
			This argument can be set if just the X component of the position is to be changed.
		y : int | float | np.int32 | np.int64 | np.float32 | np.float64 | None
			This argument can be set if just the Y component of the position is to be changed.
		z : int | float | np.int32 | np.int64 | np.float32 | np.float64 | None
			This argument can be set if just the Z component of the position is to be changed.
		**kwargs
			Keyword arguments are passed to the :meth:`copy`.
		
		Returns
		-------
		blue.LatticeType
			A shifted copy is returned.
		"""
		if pos is not None and any(arg is not None for arg in [x, y, z]):
			raise ValueError(f'Only argument (pos) or arguments (x, y, z) can be set. Method call got ({', '.join(name for name, val in [('pos', pos), ('x', x), ('y', y), ('z', z)] if val is not None)}).')
		if pos is None:
			x = float(x) if x is not None else 0.
			y = float(y) if y is not None else 0.
			z = float(z) if z is not None else 0.
			pos = np.array([x, y, z], dtype=np.float32)
		def rec(depth, field, pos, kwargs):
			if depth == 1:
				return [thing.shift(pos, **kwargs) for thing in field]
			else:
				return [rec(depth - 1, subfield, pos, kwargs) for subfield in field]
		things  = rec(self.n_dim, self._things, pos, kwargs)
		# MIRROR INIT
		lattice = object.__new__(Lattice)
		lattice._directions   = self._directions.copy()
		lattice._repetitions = self._repetitions.copy()
		lattice._thing       = self._thing.copy(**kwargs)
		lattice._things      = things
		return lattice


	def locate(self, 
		   pos: list[int|float]|np.ndarray|None                        = None, 
		   x:   int|float|np.int32|np.int64|np.float32|np.float64|None = None, 
		   y:   int|float|np.int32|np.int64|np.float32|np.float64|None = None, 
		   z:   int|float|np.int32|np.int64|np.float32|np.float64|None = None, 
		   **kwargs):
		"""
		This method creates a copy that is relocated to ``pos``.
		
		Parameters
		----------
		pos : list[int | float] | np.ndarray | blue.MoveableThingType | None
			The position to which the Thing is shifted, optionally another :class:`MoveableThing` 
			can be passed in which case the position of this Thing is taken.
		globally : bool, optional
			Determining whether the relocation is relative to the local orientation or the global orientation.
		x : int | float | np.int32 | np.int64 | np.float32 | np.float64 | None
			This argument can be set if just the X component of the position is to be changed.
		y : int | float | np.int32 | np.int64 | np.float32 | np.float64 | None
			This argument can be set if just the Y component of the position is to be changed.
		z : int | float | np.int32 | np.int64 | np.float32 | np.float64 | None
			This argument can be set if just the Z component of the position is to be changed.
		**kwargs
			Keyword arguments are passed to :meth:`copy`.
		
		Returns
		-------
		blue.LatticeType
			A relocated copy is returned.
		"""
		if pos is not None and any(arg is not None for arg in [x, y, z]):
			raise ValueError(f'Only argument (pos) or arguments (x, y, z) can be set. Method call got ({', '.join(name for name, val in [('pos', pos), ('x', x), ('y', y), ('z', z)] if val is not None)}).')
		if pos is None:
			x = float(x) if x is not None else 0.
			y = float(y) if y is not None else 0.
			z = float(z) if z is not None else 0.
			pos = np.array([x, y, z], dtype=np.float32)
		def rec(depth, field, pos, kwargs):
			if depth == 1:
				return [thing.locate(pos, **kwargs) for thing in field]
			else:
				return [rec(depth - 1, subfield, pos, kwargs) for subfield in field]
		things  = rec(self.n_dim, self._things, pos, kwargs)
		# MIRROR INIT
		lattice = object.__new__(Lattice)
		lattice._directions   = self._directions.copy()
		lattice._repetitions = self._repetitions.copy()
		lattice._thing       = self._thing.copy(**kwargs)
		lattice._things      = things
		return lattice


	@blue.restrict
	def __getitem__(self, keys: int|slice|tuple[int|slice]) -> blue.LatticeViewType|blue.MoveableThingType:
		return blue.LatticeView(self._things, self).__getitem__(keys)



	def __getattr__(self, attr: str):
		if attr in self.__RESERVED:
			return super().__getattr__(attr)
		else:
			return blue.LatticeView(self._things, self).__getattr__(attr)
	

	def __setattr__(self, attr: str, val: object):
		if attr in self.__RESERVED:
			super().__setattr__(attr, val)
		else:
			blue.LatticeView(self._things, self).__setattr__(attr, val)


	#@blue.restrict
	#def __setitem__(self, keys:int|slice|tuple[int|slice], val: list|blue.ThingType|blue.LatticeView):
	#	pass
