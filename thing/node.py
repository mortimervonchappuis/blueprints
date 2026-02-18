import blueprints as blue
from blueprints import restrict
from . import base

import numpy as np
import inspect
import xml.etree.ElementTree as xml
from copy import copy
from collections import defaultdict
from itertools import chain



class NodeThing(base.BaseThing, blue.NodeThingType):
	"""
	If an inheriting Thing class can have children attached to it in the kinematic 
	tree it derives from NodeThing. It contains the methods :meth:`NodeThing.attach` and 
	:meth:`NodeThing.detach` can be used add or detach children. 
	
	If a Thing class inherits from :class:`NodeThing` it needs to define a dictionary
	named ``_CHILDREN`` which has the names of children attributes as keys and as values 
	a second level dictionary containing the ThingType of the respective children under 
	the keys ``'type'`` and a reference to the list in which the children are stored under 
	the key ``'children'``.
	"""
	@restrict
	def _build(self, parent, world, indicies, **kwargs):
		"""
		This method is called by the Things parent to construct the xml representations of 
		the kinematic tree. It recursively calls the :meth:`BaseThing._build` methods if its 
		children.
		
		Parameters
		----------
		parent : xml.etree.ElementTree.Element
			The xml element of its parent
		world : WorldType
			The World from which the build method was called initially
		
		Returns
		-------
		xml.etree.ElementTree.Element
			The built xml element of the Thing.
		"""
		self._xml_root = xml.SubElement(parent, 
						self._MUJOCO_OBJ, 
						**self._mujoco_specs(kwargs))
		# BUILDING CHILDREN
		self._build_children(parent=self._xml_root, 
				     world=world, 
				     indicies=indicies)
		return self._xml_root

	# KINEMATIC TREE METHODS/PROPERTIES

	def __getattr__(self, attr: str) -> blue.View|object|None:
		"""
		Attributes that handle access to the children of a :class:`NodeThing` are 
		defined implicitly in :attr:`NodeThing._CHILDREN`. If the attribute name can 
		be retrieved from :attr:`NodeThing._CHILDREN` a :class:`blueprints.utils.view.View` is 
		returned.
		
		If the attribute cannot be obtained from :attr:`NodeThing._CHILDREN` the call is 
		passed to ``super().__getattr__`` if is it exists.
		
		Parameters
		----------
		attr : str
			Name of the attribute that is requested.
		
		Returns
		-------
		blue.View | object | None
			Either a View to the requested children is returned, or a value obtained from 
			``super().__getattr__``.
		
		Raises
		------
		AttributeError
			If the attribute does not exist an Error is raised.
		"""

		# THIS IS MAGIC! DON'T TOUCH
		# Reserved attributes _CHILDREN and _PSEUDO_CHILDREN might not exist always (for example
		# during init). The cunjunction with RESERVED in the boolean flag compute ensures, that 
		# no infinite recursion is triggered if getattr on _CHILDREN or _PSEUDO_CHILDREN is called.
		# BEGIN MAGIC
		RESERVED = attr not in {'_CHILDREN', '_PSEUDO_CHILDREN'}#, 'descendants', 'children'}
		CHILDREN = RESERVED and attr in self._CHILDREN
		PSEUDO_C = RESERVED and hasattr(self, '_PSEUDO_CHILDREN') and attr in self._PSEUDO_CHILDREN
		# END MAGIC
		if CHILDREN or PSEUDO_C:
			if attr in self._CHILDREN:
				 elements = self._CHILDREN[attr]['children']
			else:
				 elements = self._PSEUDO_CHILDREN[attr]['children']
			return blue.View(elements=elements, 
					 name=attr, 
					 parent=self)
		#if attr != '_CHILDREN' and attr in self._CHILDREN:
		#	elements = self._CHILDREN[attr]['children']
		#	return blue.View(elements=elements, 
		#			 name=attr, 
		#			 parent=self)
		elif hasattr(super(), '__getattr__'):
			return super().__getattr__(attr)
		else:
			raise AttributeError(f'{type(self)} does not have attribute {attr}')


	def __setattr__(self, attr: str, val: object) -> None:
		"""
		If a valid children name is set, val must be either :class:`list` or 
		:class:`blueprints.utils.view.View`. The children contained in val are 
		then attached to the Thing and the previous children attached under the 
		attribute are detached.
		
		Parameters
		----------
		attr : str
			Name of the attribute that is set.
		val : object
			The value that is to be assigned to the attribute.
		
		Raises
		------
		ValueError
			If val is not :class:`list` or :class:`blueprints.utils.view.View` an Error is raised.
		"""
		if hasattr(self, '_CHILDREN') and attr in self._CHILDREN:
			if not isinstance(val, (list, blue.ThingType, blue.View)):
				raise ValueError(f'Setting the children of a object is only possible for Thing, list, or View, got {type(val)}.')
			children = self.__getattr__(attr)
			val      = [val] if isinstance(val, blue.ThingType) else val
			self.detach(*children)
			self.attach(*val, copy=False)
		else:
			super().__setattr__(attr, val)


	@restrict
	def copy(self, 
		 shallow: bool = False, 
		 **kwargs) -> blue.NodeThingType:
		"""
		This method creates a copy of the Thing from which it was called. Additional 
		modifications to the copy can be set as keyword arguments. If shallow is True 
		only the Thing itself, otherwise its children are also copied and attached.
		
		Parameters
		----------
		shallow : bool, optional
			Indicating whether the children should be copied and attached as well.
		**kwargs
			Specifies additional attributes that are passed to the copied 
			:meth:`NodeThing.__init__`.
		
		Returns
		-------
		blue.NodeThingType
			The returned value is a copy of the :class:`ǸodeThing` itself optionally 
			including copies or its children.
		"""
		if blue.REGISTER.copy_root is None:
			blue.REGISTER.copy_root = self
		if shallow:
			children        = {name: None for name in self._CHILDREN.keys()}
			single_children = {name: None for name in self._SINGLE_CHILD_ATTR}
		elif hasattr(self, '_CHILDREN'):
			children  = {name: [thing.copy() for thing in list(child_dict['children'])] \
							 for name, child_dict in self._CHILDREN.items() \
							 if name not in kwargs and not issubclass(child_dict['type'], blue.CyclicalThingType)}
			cyclicals = {name: [thing        for thing in list(child_dict['children'])] \
							 for name, child_dict in self._CHILDREN.items() \
							 if name not in kwargs and     issubclass(child_dict['type'], blue.CyclicalThingType)}
			single_children = {name: self.__getattribute__(name).copy() if name not in self._NO_COPY_ATTR() else self.__getattribute__(name) \
										    for name in self._SINGLE_CHILD_ATTR() \
										    if self.__getattribute__(name) is not None}
			pack = lambda x: x if isinstance(x, list) else [x]
			copy = lambda x: x.copy()
			kwargs_children = {name: list(map(copy, pack(val))) for name, val in kwargs.items() if name in self._CHILDREN or name in self._SINGLE_CHILD_ATTR()}
			kwargs.update(kwargs_children)
		else:
			children        = {}
			cyclicals       = {}
			single_children = {}
		blueprint_specs = self._blueprint_specs()
		blueprint_specs.update(children)
		blueprint_specs.update(single_children)
		blueprint_specs.update(kwargs)
		if 'name' not in kwargs:
			blueprint_specs['name'] = self._name
		thing = self.__class__(copy=False, **blueprint_specs)
		if not shallow and hasattr(self, '_CHILDREN'):
			self._migrate_children(thing, cyclicals)
			#self._migrate_children(thing, children, cyclicals)
		if blue.REGISTER.copy_root is self:
			blue.REGISTER.copy_root = None
		return thing


	@restrict
	def _migrate_children(self, thing, cyclicals):
		"""
		This methods finalizes the migration for cyclical kinematic hierachies.
		
		Parameters
		----------
		thing : NodeTingType
			The Thing to which self is migrating.
		#children : dict
		#	A dictionary with all acyclical children ThingTypes in the format of 
		#	:attr:`children`.
		cyclicals : dict
			A dictionary with all cyclical children ThingTypes in the format of 
			:attr:`children`.
		"""
		for children in cyclicals.values():
			for child in children:
				child._migrate('parent', thing)
		for child_dict in self._CYCLE_REF.values():
			for cyclical in list(child_dict['children']):
				cyclical._migrate(cyclical._PARENT_REFERENCE, thing)
		for child_dict in self._CHILDREN.values():
			if issubclass(child_dict['type'], blue.FocalThingType):
				for focal in child_dict['children']:
					focal._migrate(self, thing)


	@restrict
	def attach(self, 
		   *items:   list[blue.ThingType|blue.LatticeType|blue.ViewType], 
		   globally: bool = False, 
		   copy:     bool = True) -> None:
		"""
		This method attaches ``*items`` to the parent Thing. If copy is set true, a copy of the items is created. 
		If copy is set to false, the kinematic graph might no longer be a tree resulting in infinite loops 
		or naming conflicts if the same item is included in the tree twice. All items of a specific type 
		can be accessed via the types attribute.
		
		Parameters
		----------
		*items : list[blue.ThingType]
			All items that are to be attached.
		globally : bool
			A flag specifing whether the Things are attached in the local or global position and orientation.
		copy : bool, optional
			A flag indicating whether a copy of the items should be attached. It is recommended for the user to not set this argument to false.
		
		Raises
		------
		TypeError
			If an item is not a valid child an error is raised.
		"""
		#if self._launched:
		#	raise Exception('No Things can be attached after the World has been built. Attach all Things before build or use World.unbuild().')
		views   =                   list(chain( *filter(lambda x:     isinstance(x, blue.ViewType),                     items)))
		lattice =                   list(chain( *filter(lambda x:     isinstance(x, blue.LatticeType),                  items)))
		items   = views + lattice + list(        filter(lambda x: not isinstance(x, (blue.ViewType, blue.LatticeType)), items))
		for item in items:
			if copy:
				child = item.copy()
			else:
				child = item
				if isinstance(child, blue.NodeThing):
					if child in self.path:
						raise ValueError(f'Circular Attachment. {repr(child)} is a parent of {repr(self)}. Consider setting copy=True.')
			if child.parent is not None:
				child.parent.detach(child)
			# GLOBALLY
			transform = globally and isinstance(child, blue.MoveableThingType) and isinstance(self, blue.MoveableThingType)
			if transform:
				R1  = child.global_rotation_matrix
				R2  = self.global_rotation_matrix
				R3  = R2.T @ R1
				euler = blue.Rotation.reference_frame_to_euler(R3)
				pos = child.global_pos
			types = tuple(map(lambda x: x['type'], filter(lambda x: not issubclass(x['type'], blue.TendonType), self._CHILDREN.values())))
			if not isinstance(child, types):
				raise TypeError(f'{type(self)} only takes {types} for attachment got {type(child)} instead.')
			for name, child_dict in self._CHILDREN.items():
				child_type = child_dict['type']
				children   = child_dict['children']
				if isinstance(child, child_type):
					children.append(child)
					child._parent = self
					#child.parent = self
					break
			#child._name_manager = None
			if transform:
				child.global_pos = pos
				child.euler = list(euler)


	@restrict
	def detach(self, *items: list[blue.ThingType]) -> None:
		"""
		This method is used to detach Things from the kinematic tree. They are afterwards 
		no longer present in this Things children attributes and their ``parent`` attribute 
		will be set to ``None``. 
		
		Parameters
		----------
		*items
			Things that are no longer children of this Thing can be passed, as long as their 
			type is a valid type for children of this :class:`NodeThing`.
		
		Raises
		------
		TypeError
			If the arguments to this function are not valid children types an error is raised.
		"""
		if self._launched:
			raise Exception('No Things can be detached after the World has been built. Detach all Things before build or use World.unbuild().')
		views = list(chain( *filter(lambda x:     isinstance(x, blue.ViewType), items)))
		items = views + list(filter(lambda x: not isinstance(x, blue.ViewType), items))
		for item in items:
			#types = tuple(map(lambda x: x['type'], self._CHILDREN.values()))
			#if not isinstance(item, types):
			#	raise TypeError(f'{type(self)} only takes {types} or Views of those types for detachment got {type(item)} instead.')
			# FIND THE CHILD IN CHILDREN
			for name, child_dict in self._CHILDREN.items():
				child_type = child_dict['type']
				children   = child_dict['children']
				if isinstance(item, child_type):
					if item in children:
						# REMOVE CHILD
						children.remove(item)
						#item._name_scope = None
						item._parent = None
						#item.parent = None
						# DECOUPLE ORPHANED REFERENCES OF CYCLICAL THINGS
						if isinstance(item, blue.NodeThingType):
							#item._name_manager = blue.naming.NameManager(item)
							item._decouple_descendants()
						if isinstance(item, blue.CyclicalThingType):
							item._decouple()
			# FIND REF IN CYCLE REFS
			for name, child_dict in self._CYCLE_REF.items():
				child_type = child_dict['type']
				children   = child_dict['children']
				if isinstance(item, child_type):
					if item in children:
						# REMOVE CHILD
						children.remove(item)
						#item._name_scope = None
						item.__setattr__(item._PARENT_REFERENCE, None)
		self._decouple_descendants()


	def __contains__(self, thing: blue.ThingType) -> bool:
		"""
		Parameters
		----------
		thing : ThingType
			On checking whether the thing is a child instance identity is used rather 
			than a weaker attribute equivalence.
		
		Returns
		-------
		bool
			Indicates if the thing is currently a child of this :class:`NodeThing`.
		"""
		for child_dict in self._CHILDREN.values():
			child_type = child_dict['type']
			children   = child_dict['children']
			if isinstance(thing, child_type):
				if thing in children:
					return True
		return False


	@property
	def _location_range(self) -> tuple[np.ndarray, np.ndarray]|None:
		"""
		This function is used to determine the range in which the Thing and its 
		children are located.
		
		Returns
		-------
		tuple[np.ndarray, np.ndarray]
			The returned values are two numpy arrays of shape (3,) containing the 
			minimum and maximum of the coordinate values of each dimension which 
			contain all points of the Thing including its children.
		"""
		min_values, max_values = [], []
		if isinstance(self, blue.MoveableThingType):
			min_pos, max_pos = blue.MoveableThing._location_range.fget(self)
			min_values.append(min_pos)
			max_values.append(max_pos)
		for dec_dict in self.descendants.values():
			for decendant in dec_dict['descendants']:
				if hasattr(self, '_background'):
					if decendant in self._background:
						continue
				if hasattr(decendant, 'pos') and hasattr(decendant, 'size'):
					min_pos, max_pos = decendant._location_range
					min_values.append(min_pos)
					max_values.append(max_pos)
		if min_values and max_values:
			min_pos = np.min(np.array(min_values), axis=0)
			max_pos = np.max(np.array(max_values), axis=0)
			return min_pos, max_pos
		else:
			return np.zeros(3), np.zeros(3)


	@property
	def children(self) -> dict:
		"""
		The names and lists of all types of children are specified in the :attr:`_CHILDREN` attribute 
		as well as the type. This property is a reduced dictionary only containing names and lists.
		
		Returns
		-------
		dict
			A dictionary containing all children of the NodeThing. 
		"""
		return {name: child['children'] for name, child in self._CHILDREN.items()}


	@property
	def descendants(self) -> dict:
		"""
		Nested dictionaries — The top level contains the names of the respective 
		descendants property names, on the second level there are the keys ``'type'`` 
		and ``'descendant'`` containing the ThingType and the descendant of the attribute.
		
		Returns
		-------
		dict
			The structure of descendants differs from :attr:`children`.
		"""
		descendants = defaultdict(lambda: {'type':        None, 
						   'descendants': list()})
		# SINGLE CHILDREN
		queue = list((self, key, val) for key, val in self._SINGLE_CHILD_ATTR().items())
		while queue:
			thing, name, child_type = queue.pop()
			single = thing.__getattribute__(name)
			if single is None:
				continue
			descendants[name]['type'] = child_type
			descendants[name]['descendants'].append(single)
			queue += list((single, key, val) for key, val in single._SINGLE_CHILD_ATTR().items())
		# CHILDREN
		for name, child_dict in self._CHILDREN.items():
			child_type = child_dict['type']
			children   = child_dict['children']
			descendants[name]['type'] = child_type
			descendants[name]['descendants'].extend(children)
			if issubclass(child_type, blue.NodeThingType):
				for child in children:
					for dec_name, dec_dict in child.descendants.items():
						dec_type     = dec_dict['type']
						dec_children = dec_dict['descendants']
						descendants[dec_name]['type'] = dec_type
						#for dec_child in dec_children:
						#	if dec_child not in descendants[dec_name]['descendants']:
						#		descendants[dec_name]['descendants'].append(dec_child)
						descendants[dec_name]['descendants'].extend(dec_children)
		# ELIMINATE DUPLICATES
		for dec_name, dec_dict in descendants.items():
			dec_list = dec_dict['descendants']
			dec_dict['descendants'] = [x for i, x in enumerate(dec_list) if dec_list.index(x) == i]
		return descendants


	@property
	def all(self) -> blue.AllViewType:
		"""
		This property gives access to a :class:`View <blueprints.utils.view.AllView>` 
		through which attributes of all descendants can be handled. See :class:`AllView <blueprints.utils.view.AllView>` for details.

		Returns
		-------
		AllViewType
			A handle for accessing attributes of all descendants of the NodeThing.
		"""
		return blue.AllView(self.descendants, self)


	def _check_children_types(self):
		"""
		This method should be called at the end of a Thing class deriving from :class:`NodeThing` 
		:meth:`__init__`. It checks is all children have the proper type.
		
		Raises
		------
		TypeError
			If a child does not have the correct type and error is raised.
		"""
		for child_dict in self._CHILDREN.values():
			child_type = child_dict['type']
			children   = child_dict['children']
			if issubclass(child_type, blue.BodyType):
				for child in children:
					if type(child) == blue.Agent:
						raise TypeError(f"""A type restriction in function {self.__init__.__name__} was violated: 
An argument for bodies is was of type Agent. 
The received argument for {child.__class__.__name__.lower()} was {children}.""")
			for child in children:
				if not isinstance(child, child_type):
					raise TypeError(f"""A type restriction in function {self.__init__.__name__} was violated: 
{child.__class__.__name__.lower()} is supposed to be of type {child_type}.
The received argument for {child.__class__.__name__.lower()} was {children}.""")


	@restrict
	def _build_children(self, parent, world, indicies, exclude=None) -> None:
		"""
		This methods build the children for xml construction. If some children should 
		be excluded from the build, the names of their attribute can be passed in exclude.
		
		Parameters
		----------
		parent : xml.etree.ElementTree.Element
			The xml element of the childrens parent.
		world : WorldType
			The world from which the build method was called initially.
		exclude : None, optional
			Either a single string or an iterable of strings or the names of children 
			attributes which should be excluded from the build.
		"""
		if exclude is None:
			exclude = ()
		elif isinstance(exclude, str):
			exclude = (exclude,)
		# BUILD CHILDREN
		for name, child_dict in self._CHILDREN.items():
			children = child_dict['children']
			if name not in exclude:
				self._build_tree(children=children, 
						 parent=parent, 
						 world=world, 
						 indicies=indicies)


	@restrict
	def _build_tree(self, 
			children: list[blue.ThingType], 
			parent, 
			world, 
			indicies) -> None:
		"""
		This method build all children from a children attribute.
		
		Parameters
		----------
		children : list[blue.ThingType]
			A list of the children to be build.
		parent : xml.etree.ElementTree.Element
			The xml element of the childrens parent.
		world : WorldType
			The world from which the build method was called initially.
		"""
		for child in children:
			child._build(parent=parent, 
				     world=world, 
				     indicies=indicies)


	def _decouple_descendants(self):
		"""
		This method is used to decouple references of :class:`CyclicalThing` that 
		no longer point inside the kinematic tree. This can be the result of the 
		removal of a Thing. Decoupled references are set to None.
		"""
		for dec_dict in self.descendants.values():
			dec_type = dec_dict['type']
			dec_list = dec_dict['descendants']
			if issubclass(dec_type, blue.CyclicalThingType):
				for dec in dec_list:
					dec._decouple()
