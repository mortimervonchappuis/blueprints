"""
If the children attributes of a :class:`NodeThing <blueprints.thing.node.NodeThing>` are called, those children will be 
returned in a :class:`View`. The :class:`View` can be used to get and set attributes of multiple objects 
as well as to call methods on those child Things simultaneously. The following example shows how to build 
a simplified hand model using Views.

.. code-block:: python
	:caption: Getting attributes
		
	>>> names = ['thumb', 'index', 'middle', 'ring', 'pinky']
	>>> fingers = [blue.Body(name=name) for name in names]
	>>> palm = blue.Body(name='palm', bodies=fingers)
	>>> view = palm.bodies # calling child attributes creates a view
	>>> view
	View[5|palm.bodies]
	>>> view.name
	['thumb', 'index', 'middle', 'ring', 'pinky']
	>>> view.bodies
	View[0|palm.bodies.bodies]
	>>> intermediate = blue.Body(name='intermediate')
	>>> distal = blue.Body(name='distal')
	>>> phalanges = [distal, intermediate, intermediate, intermediate, intermediate]
	>>> view.attach.distribute(phalanges) 
	>>> view.bodies[1:].attach(blue.Body(name='distal'))

.. code-block:: python
	:caption: Setting attributes
		
	>>> view.bodies[0].name = 'distal_thumb'
	>>> view.bodies[1:].name = ['intermediate_index', 
				    'intermediate_middle', 
				    'intermediate_ring', 
				    'intermediate_pinky']
	>>> view.bodies.name
	['distal_thumb', 
	 'intermediate_index', 
	 'intermediate_middle',
	 'intermediate_ring', 
	 'intermediate_pinky']

.. code-block::
	:caption: Applying functions
		
	>>> renaming = lambda x: x.__setattr__('name', f'distal_{x.parent.parent.name}')
	>>> view.bodies.bodies.apply(renaming)
	>>> view.bodies.bodies.name
	['distal_index', 'distal_middle', 'distal_ring', 'distal_pinky']

.. code-block:: mxml
	:caption: XML structure

	<body name="palm">
	    <body name="thumb">
	        <body name="distal_thumb" />
	    </body>
	    <body name="index">
	        <body name="intermediate_index">
	            <body name="distal_index" />
	        </body>
	    </body>
	    <body name="middle">
	        <body name="intermediate_middle">
	            <body name="distal_middle" />
	        </body>
	    </body>
	    <body name="ring">
	        <body name="intermediate_ring">
	            <body name="distal_ring" />
	        </body>
	    </body>
	    <body name="pinky">
	        <body name="intermediate_pinky">
	            <body name="distal_pinky" />
	        </body>
	    </body>
	</body>
"""

import numpy as np
import blueprints as blue
from collections import defaultdict
from itertools import count, chain
from types import BuiltinFunctionType, FunctionType, MethodType



class View(blue.ViewType):
	"""
	The View class is used to handle the access to children of nodes in a kinematic tree and their properties.
	"""
	__RESERVED = {'_View__ELEMENTS', 
		      '_View__NAME', 
		      '_View__PARENT', 
		      '_View__RESERVED', 
		      '_AllView__ELEMENTS', 
		      '_AllView__NAME', 
		      '_AllView__PARENT', 
		      '_AllView__RESERVED', 
		      '_AllView__DESCENDANTS'}
	
	@blue.restrict
	def __init__(self, elements: list, name: str, parent: blue.ThingType|blue.LatticeType) -> None:
		"""
		The View can be used to get and set attributes of multiple objects at the same time.
		
		Parameters
		----------
		elements : list
			All Things, that are part of the View.
		name : str
			The name of the attribute that was requested for this View.
		parent : blue.ThingType
			The parent of the View.
		"""
		# ELIMINATE DUPLICATES
		elements = [x for i, x in enumerate(elements) if elements.index(x) == i]
		self.__ELEMENTS = elements.copy()
		self.__NAME     = name
		self.__PARENT   = parent


	def __repr__(self) -> str:
		"""
		Represents the View as a string.
		
		Returns
		-------
		str
			The representing string contains a representation of the call that was used to get the View.
		"""
		return f'View[{len(self)}|{self.__PARENT.name}.{self.__NAME}]'


	def __len__(self):
		"""
		The number of Things contained in the View.
		
		Returns
		-------
		int
			The number of Things contained in the View.
		"""
		return len(self.__ELEMENTS)


	def __iter__(self):
		"""
		Used to iterate over all Things in the View. While iterating over the View, its contents should not be modified.
		
		Yields
		------
		ThingType
			The next Thing in the list of elements.
		"""
		for element in self.__ELEMENTS:
			yield element


	def __bool__(self):
		"""
		Indicating whether the View contains elements.
		
		Returns
		-------
		bool
			If the View is empty the resulting value is False and True otherwise.
		"""
		return len(self.__ELEMENTS) > 0


	@blue.restrict
	def __getitem__(self, key: int|slice|str) -> blue.ThingType|blue.ViewType:
		"""
		This method either returns a single Thing from the View or a new View from a subset of the View.
		
		Parameters
		----------
		key : int | slice | str
			If the argument is an integer, the Thing with its corresponding index is returned.
			If the argument is a slice, a View containing the Things according to the slices indicies is returned.
			if the argument is a string, a View containing the Things with the same user defined names is returned.
		
		Returns
		-------
		blue.ThingType | blue.ViewType
		
		Raises
		------
		NotImplemented
			If the argument is neither (int, slice, str) NotImplemented is raised.
		"""
		if isinstance(key, int):
			return self.__ELEMENTS[key]
		elif isinstance(key, slice):
			return View(self.__ELEMENTS[key], self.__NAME, self.__PARENT)
		elif isinstance(key, str):
			return View(list(filter(lambda x: x._name == key, self)), self.__NAME, self.__PARENT)
		else:
			raise NotImplemented


	def __getattr__(self, attr: str) -> list|blue.ViewType|blue.FunctionHandleType:
		"""
		This methods handles the access to the attributes of the Things contained in the View 
		depending on the type of attribute:
		1. *property* In this case a list of all the singularly retrieved attributes is returned.
		2. *children* This returns a new View, that contains the children of all the Views Things.
		3. *method/function* In this case a FunctionHandle is returned. 
		
		In the last case the :class:`FunctionHandle` can either be called directly and all arguments are 
		passed to each of the Views Thing or the distribute method of the FunctionHandle can be used, 
		to match each argument to each Thing in a one to one relation.
		
		.. code-block::
			:caption: Example
		
			>>> names   = ['thumb', 'index', 'middle', 'ring', 'pinky']
			>>> fingers = [blue.Body(name=name) for name in names]
			>>> palm = blue.Body(name='palm', bodies=fingers)
			>>> view = palm.bodies # calling child attributes creates a view
			>>> view
			View[5|palm.bodies]
			>>> view.name # CASE 1
			['thumb', 'index', 'middle', 'ring', 'pinky']
			>>> view.bodies # CASE 2
			View[0|palm.bodies.bodies]
			>>> intermediate = blue.Body(name='intermediate')
			>>> distal = blue.Body(name='distal')
			>>> phalanges = [distal, intermediate, intermediate, intermediate, intermediate]
			>>> view.attach.distribute(phalanges) # CASE 3
			>>> view.bodies[1:].attach(blue.Body(name='distal')) # CASE 3
		
		.. code-block:: mxml
			:caption: XML structure of the previous example
		
			<body name="palm">
			    <body name="thumb">
			        <body name="distal_(0)" />
			    </body>
			    <body name="index">
			        <body name="intermediate_(0)">
			            <body name="distal_(1)" />
			        </body>
			    </body>
			    <body name="middle">
			        <body name="intermediate_(1)">
			            <body name="distal_(2)" />
			        </body>
			    </body>
			    <body name="ring">
			        <body name="intermediate_(2)">
			            <body name="distal_(3)" />
			        </body>
			    </body>
			    <body name="pinky">
			        <body name="intermediate_(3)">
			            <body name="distal_(4)" />
			        </body>
			    </body>
			</body>
		
		
		Parameters
		----------
		attr : str
			All attributes are valid, aslong as they are properties or methods of the Thing type contained in the View.
		
		Returns
		-------
		list | blue.ViewType | blue.FunctionHandleType
		
		Raises
		------
		AttributeError
			If the attribute cannot be retrieved an Error is raised.
		"""
		if len(self) == 0:
			raise AttributeError(f'{self} is empty, attribute "{attr}" cannot be retrieved.')
		if all(hasattr(x, attr) for x in self):
			elements = list(map(lambda x: getattr(x, attr), self))
			if all(map(lambda x: isinstance(x, blue.ViewType), elements)):
				elements = list((item for view in elements for item in view))
				return View(elements, f'{self.__NAME}.{attr}', self.__PARENT)
			elif all(map(lambda x: isinstance(x, blue.ThingType), elements)):
				return View(elements, f'{self.__NAME}.{attr}', self.__PARENT)
			elif all(map(lambda x: isinstance(x, (FunctionType, MethodType)), elements)):
				return FunctionHandle(elements, f'{self.__NAME}.{attr}', self.__PARENT)
			elif all(map(lambda x: x is None, elements)):
				return [None] * len(elements) # []
			else:
				return elements
		else:
			try:
				#elements = list((item for view in map(lambda x: x.__getattr__(attr), self) for item in view))
				elements = list((item for view in map(lambda x: x.__getattribute__(attr), self) for item in view))
			except AttributeError as error:
				raise AttributeError(f'{self}.{attr} cannot be retrieved.') from None
			return View(elements, f'{self.__NAME}.{attr}', self.__PARENT)


	@blue.restrict
	def __setattr__(self, attr: str, val: object) -> None:
		"""
		This method passes the setting of attributes to each individual Thing contained in View.
		Depending on whether ``val`` is a list/View containing values for each Thing individually
		or a singular value, ``val`` is either unrolled along the first axis or given fully for each 
		assignment.

		>>> # keeping the view object from __getattr__
		>>> view.bodies[1:].name = ['intermediate_index', 
					    'intermediate_middle', 
					    'intermediate_ring', 
					    'intermediate_pinky']
		>>> view.bodies.name
		['intermediate_index', 
		 'intermediate_middle', 
		 'intermediate_ring', 
		 'intermediate_pinky']

		.. code-block:: mxml
			:caption: XML structure
		
			<body name="palm">
			    <body name="thumb">
			        <body name="distal_(0)" />
			    </body>
			    <body name="index">
			        <body name="intermediate_index">
			            <body name="distal_(1)" />
			        </body>
			    </body>
			    <body name="middle">
			        <body name="intermediate_middle">
			            <body name="distal_(2)" />
			        </body>
			    </body>
			    <body name="ring">
			        <body name="intermediate_ring">
			            <body name="distal_(3)" />
			        </body>
			    </body>
			    <body name="pinky">
			        <body name="intermediate_pinky">
			            <body name="distal_(4)" />
			        </body>
			    </body>
			</body>
		
		.. note::
			If values should be unrolled over all Things, ``vals`` must have the same length as the View!
		
		Parameters
		----------
		attr : str
			The name of the attribute to be set.
		val : object
			Either a list View or other iterable that contains different values for all children or one value of any type that is set for all.
		"""
		if attr in self.__RESERVED:
			return super().__setattr__(attr, val)
		else:
			view_attr = self.__getattr__(attr)
			for element, elem_val in zip(self, self.__nesting_value(view_attr, val)):
				element.__setattr__(attr, elem_val)


	def __delattr__(self, attr):
		"""
		This method detaches the attributes and deletes them afterwards
		"""
		view_attr = self.__getattr__(attr)
		if isinstance(view_attr, blue.ViewType):
			view_attr.parent.detach.distribute(view_attr)
			self.__PARENT._decouple_descendants()


	def __delitem__(self, key):
		"""
		This method detaches the attributes and deletes them afterwards
		"""
		view_key = self[key]
		if isinstance(view_key, blue.ViewType):
			view_key.parent.detach.distribute(view_key)
			view_key.parent._decouple_descendants()
			self.__PARENT._decouple_descendants()


	def apply(self, function, *args, **kwargs) -> None|list:
		"""
		Applies a function to all Things contained in View. Arguments and keyword arguments
		are passed to each individual call in an all to all relation.
		
		>>> # keeping the view object from __setattr__
		>>> renaming = lambda x: x.__setattr__('name', f'distal_{x.parent.parent.name}')
		>>> view.bodies.bodies.apply(renaming)
		>>> view.bodies.bodies.name
		['distal_index', 'distal_middle', 'distal_ring', 'distal_pinky']
		
		Parameters
		----------
		function : Callable
			Functions methods and classes are valid arguments.
		*args
			Arguments passed to each individual call.
		**kwargs
			Kexword arguments passed to each individual call.
		
		Returns
		-------
		None | list
			If any individual call results in a value that is not None, a list of all returned values gets returned, None otherwise.
		"""
		results = [function(element) for element in self]
		if any(x is not None for x in results):
			return results


	def __nesting_value(self, view_attr, val):
		"""
		Thie method is used to create an iterator for value assignment that either creates 
		a many to one or a one to one relation, depending on the new values shape and the 
		current attributes shape.
		"""
		view_structure = self.__nesting_structure(view_attr)
		val_structure  = self.__nesting_structure(val)
		if isinstance(val, np.ndarray) and  self.__compatible_numpy(view_structure, val):
			return np.rollaxis(val, axis=0)
		elif self.__compatible_nesting(view_structure, val_structure):
			return val
		elif isinstance(val, blue.ViewType):
			if len(val) != 1:
				raise ValueError(f'To set a View attribute to another View both sizes must be identical or the other Views size must be one, got sizes {len(view_structure)} and {len(val_structure)}.')
			else:
				return (val[0] for _ in range(len(self)))
		else:
			return (val for _ in range(len(view_structure)))



	@classmethod
	def __nesting_structure(cls, element):
		"""
		This method retrieves a representation of the elements nesting structure.
		"""
		if isinstance(element, np.ndarray):
			return element.shape
		elif isinstance(element, str):
			return str
		elif isinstance(element, View):
			return cls.__nesting_structure(list(element))
		elif hasattr(element, '__iter__'):
			return element.__class__(map(cls.__nesting_structure, element))
		else:
			return element.__class__


	@staticmethod
	def __compatible_nesting(view_structure, val_structure):
		"""
		This method determines whether the two nesting structures are compatible for a 
		one to one assignment relation for lists and other non numpy iterables.
		"""
		val_class = val_structure if isinstance(val_structure, type) else val_structure.__class__
		if not hasattr(val_structure, '__len__') or issubclass(val_class, str):
			return False
		elif len(val_structure) != len(view_structure):
			return False
		elif (not hasattr(val_structure, '__contains__') or view_structure not in val_structure) and val_structure != view_structure:
			return False
		else:
			return True


	@staticmethod
	def __compatible_numpy(view_structure, val):
		"""
		This method determines whether the two nesting structures are compatible for a 
		one to one assignment relation for numpy arrays.
		"""
		if val.shape[0] == len(view_structure) and val.ndim > 1:
			if all(isinstance(view, tuple) and view == val.shape[1:] for view in view_structure):
				return True
			elif all(isinstance(view, (list, tuple)) and len(view) == val.shape[1] for view in view_structure):
				return True
		return False



class FunctionHandle(blue.FunctionHandleType):

	"""
	This class is used to handle access to a function or method called from a :class:`View`. 

	>>> palm.bodies
	View[5|palm.bodies]
	>>> palm.bodies.shift
	ViewFunction[5|palm.bodies.shift]
	
	Arguments given to the function or method call are passed to each individual Thing
	in a many to one relation.

	>>> palm.bodies.pos
	[array([0., 0., 0.], dtype=float32), 
	 array([0., 0., 0.], dtype=float32), 
	 array([0., 0., 0.], dtype=float32), 
	 array([0., 0., 0.], dtype=float32), 
	 array([0., 0., 0.], dtype=float32)]
	>>> many_to_one = palm.bodies.shift([2, 0, -1])
	>>> many_to_one
	View[5|palm.bodies.shift(...)]
	>>> many_to_one.pos
	[array([ 2.,  0., -1.], dtype=float32), 
	 array([ 2.,  0., -1.], dtype=float32), 
	 array([ 2.,  0., -1.], dtype=float32), 
	 array([ 2.,  0., -1.], dtype=float32), 
	 array([ 2.,  0., -1.], dtype=float32)]

	If the :meth:`distributed` over all Things contained in the parenting
	View in a one to one relation by using the distribute submethod.

	>>> one_to_one = palm.bodies.shift.distribute([[0, 0, 1], 
						       [0, 1, 0], 
						       [0, 1, 1], 
						       [1, 0, 0], 
						       [1, 0, 1]])
	>>> one_to_one.pos
	[array([0., 0., 1.], dtype=float32), 
	 array([0., 1., 0.], dtype=float32), 
	 array([0., 1., 1.], dtype=float32), 
	 array([1., 0., 0.], dtype=float32), 
	 array([1., 0., 1.], dtype=float32)]

	If the result of all method or function calls are None, None is returned.

	>>> palm.bodies.attach()
	None

	If all results are Things, a View of them is returned. This allows for 
	functions and methods that return Things to be chained.

	>>> palm.bodies.copy()
	View[5|palm.bodies.copy(...)]
	>>> palm.bodies.shift([0, 0, 3]).rotate(alpha=PI).copy()
	View[5|palm.bodies.shift(...).rotate(...).copy(...)]

	Otherwise a list of all individually returned values is returned. 

	>>> palm.bodies._mujoco_specs()
	[{'name': 'thumb'}, 
	 {'name': 'index'}, 
	 {'name': 'middle'}, 
	 {'name': 'ring'}, 
	 {'name': 'pinky'}]

	"""
	
	def __init__(self, functions: list, name: str, parent: blue.ThingType) -> None:
		"""
		The FunctionHandle takes the functions or methods to be called, the name by which 
		it was called and the initial parent from which the first :class:`View` was called.
		
		Parameters
		----------
		functions : list
			All functions or methods from the Views Things.
		name : str
			The name of the called function or method.
		parent : blue.ThingType
			The parent from which the first View was called.
		"""
		self.__NAME      = name
		self.__PARENT    = parent
		self.__FUNCTIONS = functions


	def __repr__(self) -> str:
		"""
		Represents the View as a string.
		
		Returns
		-------
		str
			The representing string contains a representation of the call that was used to get the View.
		"""
		return f'ViewFunction[{len(self)}|{self.__PARENT.name}.{self.__NAME}]'


	def __iter__(self):
		"""
		Yields
		------
		Callable
			The functions or methods Things that were requested from the View.
		"""
		for functions in self.__FUNCTIONS:
			yield functions


	def __len__(self) -> int:
		"""
		Returns
		-------
		int
			The number of 
		"""
		return len(self.__FUNCTIONS)


	def __call__(self, *args, **kwargs) -> None|list:
		"""
		The call is redirected to each function or method in the FunctionHandle.
		All arguments are passed on all calls in a many to one relation.
		
		Parameters
		----------
		*args
			The arguments to be passed to the call.
		**kwargs
			The keywordarguments to be passed to the call.
		
		Returns
		-------
		None | list | View
			The aggregated result of each call.
		"""
		result = [functions(*args, **kwargs) for functions in self]
		return self._convert_result(result)


	def distribute(self, *args, **kwargs) -> None|list:
		"""
		The call is redirected to each function or method in the FunctionHandle.
		The arguments are unrolled along the first axis and distributed over all calls 
		in a one to one relation.
		
		Parameters
		----------
		*args
			The arguments to be passed to the call.
		**kwargs
			The keywordarguments to be passed to the call.
		
		Returns
		-------
		None | list | View
			The aggregated result of each call.
		"""
		kw_list = [dict(zip(kwargs.keys(), vals)) for vals in zip(*kwargs.values())]
		if args and kwargs:
			assert all(len(self) == l for l in map(len, args))
			assert len(self) == len(kw_list)
			result = [function(*arg, **kwarg) for function, arg, kwarg in zip(self, zip(*args), kw_list)]
		elif args:
			assert all(len(self) == l for l in map(len, args))
			result = [function(*arg) for function, arg in zip(self, zip(*args))]
		elif kwargs:
			assert len(self) == len(kw_list)
			result = [function(**kwarg) for function, kwarg in zip(self, kw_list)]
		else:
			result = [function() for function in self]
		return self._convert_result(result)


	def _convert_result(self, result):
		if any(x is not None for x in result):
			if all(isinstance(x, blue.ThingType) for x in result):
				return View(result, f'{self.__NAME}(...)', self.__PARENT)
			else:
				return result
		else:
			return None



class AllView(blue.AllViewType, View):
	"""
	The AllView is analogous to the :class:`View` for all descendants instead of children.
	Attribute getting and setting is handeled for all valid descendants. A ThingsType 
	attribute is then converted back to a normal view.

	.. code-block:: mxml
		:caption: XML Structure:

		<body name="A">
                        <body name="B" />
                        <geom mass="42.0" type="sphere" name="C" />
                </body>

	.. code-block:: python
		:caption: Children and Descendants View Comparison:

		>>> world.bodies
		View[1|anonymous_model.bodies]
		>>> world.bodies.name
		['A']
		>>> world.all
		View[3|anonymous_model.all]
		>>> world.all.name
		['A', 'B', 'C']
		>>> world.all.bodies
		View[2|anonymous_model.all.bodies]
		>>> world.bodies.name
		['A', 'B']

	If only some descendants have a certain attribute, the view restricts to this valid subset. 
	For example, only :class:`Geoms <blueprints.geoms.BaseGeom>` have mass, so retrieving ``all.mass`` excludes all non-Geoms.

	.. code-block:: python
		:caption: Children and Descendants View Comparison:

		>>> world.all.mass
		[42.0]
	"""
	__RESERVED = {'__ELEMENTS', 
		      '__NAME', 
		      '__PARENT', 
		      '__RESERVED', 
		      '__DESCENDANTS', 
		      '_View__ELEMENTS', 
		      '_View__NAME', 
		      '_View__PARENT', 
		      '_View__RESERVED', 
		      '_AllView__ELEMENTS', 
		      '_AllView__NAME', 
		      '_AllView__PARENT', 
		      '_AllView__RESERVED', 
		      '_AllView__DESCENDANTS'}
	@blue.restrict
	def __init__(self, descendants: dict, parent: blue.ThingType) -> None:
		"""
		Parameters
		----------
		descendants : dict
			The descendants property invoked on ``parent``
		parent : blue.ThingType
			The parent Thing from which ``.all`` was retrieved.
		"""
		elements = []
		for desc_dict in descendants.values():
			elements.extend(desc_dict['descendants'])
		super().__init__(elements=elements, name='all', parent=parent)
		self.__DESCENDANTS = descendants


	def __getattr__(self, attr: str):
		if attr in self.__RESERVED:
			return super().__getattribute__(attr)
		if attr in self.__DESCENDANTS:
			return View(self.__DESCENDANTS[attr]['descendants'], f'{self._View__NAME}.{attr}', self._View__PARENT)
		else:
			if not all(hasattr(thing, attr) for thing in self):
				elements = list(filter(lambda x: hasattr(x, attr), self))
				view = View(elements, name=self._View__NAME, parent=self._View__PARENT)
				return view.__getattr__(attr)
			else:
				return super().__getattr__(attr)


	@blue.restrict
	def __setattr__(self, attr: str, val: object) -> None:
		super().__setattr__(attr, val)
		if attr in self.__RESERVED or all(hasattr(thing, attr) for thing in self):
			super().__setattr__(attr, val)
		else:
			elements = list(filter(lambda x: hasattr(x, attr), self))
			view = View(elements, name=self._View__NAME, parent=self._View__PARENT)
			view.__setattr__(attr, val)



class LatticeView(blue.LatticeViewType):
	"""
	A LatticeView is created whenever a :class:`Lattice <blueprints.utils.lattice.Lattice>` is indexed. 
	Attribute getter and setter are handled through this View similar to :class:`View`. For details have 
	a look at :class:`Lattice <blueprints.utils.lattice.Lattice>`.
	"""
	__RESERVED = {'_LatticeView__THINGS', 
		      '_LatticeView__KEYS', 
		      '_LatticeView__KEY_STR', 
		      '_LatticeView__PARENT'}
	def __init__(self, 
		     things: list, 
		     parent: blue.LatticeType, 
		     keys:   tuple[int|slice] = ()):
		"""
		Parameters
		----------
		things : list
			The indexed Things of the parent Lattice
		parent : blue.LatticeType
			The Lattice which was indexed to create the LatticeView
		keys : tuple[int | slice]
			The keys used to index the Lattice
		"""
		def parse(key):
			if isinstance(key, int):
				return str(key)
			elif isinstance(key, slice):
				indecies = []
				if key.step is not None:
					indecies.insert(0, str(key.step))
				if key.stop is not None:
					indecies.insert(0, str(key.stop))
				else:
					indecies.insert(0, '')
				if key.start is not None:
					indecies.insert(0, str(key.start))
				else:
					indecies.insert(0, '')
				return ':'.join(indecies)
			else:
				raise NotImplemented
		self.__KEY_STR = ', '.join(map(parse, keys))
		self.__KEYS    = keys
		self.__THINGS  = things if isinstance(things, list) else [things]
		self.__PARENT  = parent


	@property
	def n_dim(self) -> int:
		"""
		The number of dimensions in the Lattice
		
		Returns
		-------
		int
		"""
		things = self.__THINGS
		n = 0
		while isinstance(things, list):
			n += 1
			things = things[0]
		return n


	@blue.restrict
	def __getitem__(self, keys: int|slice|tuple[int|slice]) -> blue.LatticeViewType|blue.MoveableThingType:
		if isinstance(keys, tuple) and len(keys) > self.n_dim:
			raise IndexError(f'Lattice {self} has {self.n_dim} axes but {len(keys)} were indexed.')
		if isinstance(keys, (int, slice)):
			keys = (keys,)
		def rec(things, keys):
			if len(keys) == 0:
				return things
			else:
				key, keys = keys[0], keys[1:]
				if isinstance(key, int):
					return rec(things[key], keys)
				else:
					return [rec(thing, keys) for thing in things[key]]
		if all(isinstance(key, int) for key in keys) and len(keys) == self.n_dim:
			return rec(self.__THINGS, keys)
		return blue.LatticeView(rec(self.__THINGS, keys), self.__PARENT, keys)


	def __iter__(self) -> blue.ThingType:
		queue = list(self.__THINGS)
		while queue:
			item = queue.pop(0)
			if isinstance(item, list):
				queue = list(item) + queue
			else:
				yield item


	def __len__(self) -> int:
		"""
		Returns
		-------
		int
			The number of Things contained in the LatticeView.
		"""
		things = self.__THINGS
		l = 1 if things else 0
		while isinstance(things, list):
			l *= len(things)
			things = things[0]
		return l


	def __repr__(self) -> str:
		"""
		Represents the View as a string.
		
		Returns
		-------
		str
			The representing string contains a representation of the call that was used to get the View.
		"""
		if self.__KEY_STR:
			return f'LatticeView[{self.__KEY_STR}]'
		else:
			return 'LatticeView'


	def __getattr__(self, attr):
		if len(self) == 0:
			raise AttributeError(f'{self} is empty, attribute "{attr}" cannot be retrieved.')
		if all(hasattr(x, attr) for x in self):
			things = self._rec(lambda x: getattr(x, attr), list, self.__THINGS)
			if self._rec(lambda x: isinstance(x, blue.ViewType), all, things):
				things = list(self._rec(list, lambda x: list(chain(*x)), things))
				return View(things, attr, self.__PARENT)
			elif self._rec(lambda x: isinstance(x, blue.ThingType), all, things):
				things = list(self._rec(lambda x: [x], lambda x: list(chain(*x)), things))
				return View(things, attr, self.__PARENT)
			elif self._rec(lambda x: isinstance(x, (FunctionType, MethodType)), all, things):
				things = list(self._rec(lambda x: [x], lambda x: list(chain(*x)), things))
				return FunctionHandle(things, attr, self.__PARENT)
			elif self._rec(lambda x: x is None, all, things):
				return [None] * len(self) # []
			else:
				return things
		else:
			try:
				things = self._rec(lambda x: x.__getattribute__(attr), list, self.__THINGS)
			except AttributeError as error:
				raise AttributeError(f'{self}.{attr} cannot be retrieved.') from None
			return View(things, attr, self.__PARENT)


	def __setattr__(self, attr, value):
		if attr in self.__RESERVED:
			super().__setattr__(attr, value)
		else:
			view_attr = self.__getattr__(attr)
			view_structure = self.__nesting_structure(view_attr)
			val_structure  = self.__nesting_structure(value)
			if view_structure == val_structure:
				def rec(func, things, values):
					if isinstance(things, list):
						for thing, value in zip(things, values):
							rec(func, thing, value)
					else:
						func(things, values)
				rec(lambda thing, value: thing.__setattr__(attr, value), self.__THINGS, value)
			else:
				def rec(func, things):
					if isinstance(things, list):
						for thing in things:
							rec(func, thing)
					else:
						func(things)
				rec(lambda thing: thing.__setattr__(attr, value), self.__THINGS)


	@classmethod
	def _rec(cls, func, aggregate, things):
		"""
		This classmethod is a helper to apply a function on individual Things and aggregate the results. 
		"""
		if isinstance(things, list):
			return aggregate(cls._rec(func, aggregate, thing) for thing in things)
		else:
			return func(things)


	@classmethod
	def __nesting_structure(cls, element):
		"""
		This method retrieves a representation of the elements nesting structure.
		"""
		if isinstance(element, np.ndarray):
			return element.shape
		elif isinstance(element, str):
			return str
		elif isinstance(element, View):
			return cls.__nesting_structure(list(element))
		elif hasattr(element, '__iter__'):
			return element.__class__(map(cls.__nesting_structure, element))
		else:
			return element.__class__
