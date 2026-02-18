"""
.. note::
	If you found yourself here not being a dev, you should probably :attr:`skip this section <BaseThing.ID>` or 
	look up the correspinging docs page of the Thing you were looking after for which this module only 
	implements its corresponding mirror types. A link to the docs page is to be found in the Types doc 
	string.
"""


import blueprints as blue
from blueprints import restrict

import numpy as np
import inspect
import xml.etree.ElementTree as xml
from copy import copy
from collections import defaultdict



class BaseThing(blue.ThingType):

	"""
	BaseThing implements the basic attributes that are used by all Things. 
	
	All Things have certain attributes that are included in the mujoco XML construction. 
	The names of those attributes can be obtained from the :attr:`_MUJOCO_ATTR` property 
	which is dictionary with attribute names as keys and the types their value will have 
	in the Thing instance. If a new Thing class is defined that inherits from BaseThing that 
	introduces new attributes to be written to xml, they can be added by defining a class 
	property ``_NEW_MUJOCO_ATTR`` in the same structure that ``_MUJOCO_ATTR`` is defined.
	``_MUJOCO_ATTR`` then simply aggregates the new additions from ``_NEW_MUJOCO_ATTR``. 
	If a new Thing class inherits a mujoco attribute, that is no longer used, it can be 
	deleted from ``_MUJOCO_ATTR`` by defining it in a class property ``_DEL_MUJOCO_ATTR``, 
	must be a set of the names to be detached.
	
	When a Thing is copied, all attributes needed to reconstruct it are retrieved to create
	a new instance. For this the :attr:`blueprints.type.ThingType._BLUEPRINT_ATTR` property is used. New attributes 
	that are feed into the inheriting ``Thing.__init__`` are registered in the class property 
	``_NEW_BLUEPRINTS_ATTR`` and old attributes that are no longer used are registered in 
	``_DEL_BLUEPRINTS_ATTR``. Their structures are analogous to the ``_MUJOCO_ATTR``.
	
	If a Thing gets copied all attributes necessary to instantiate the copy are themselves 
	copied to ensure that the Thing and its copy do not share a reference to the same object
	in any attribute. However, if an attribute is added to :attr:`blueprints.type.ThingType._NO_COPY_VALS` it is not 
	copied. New attributes to be excluded from being copied can be added by setting the class
	property ``_NEW_NO_COPY_VALS`` to the classes Type as a set of attribute names.
	
	If a Thing is reconstructed from an xml string, some attribute in the xml tag are not 
	attributes of the Thing class. For example, a geom xml tag will have a type defining
	its shape ``<geom type='sphere'>`` but the corresponding :class:`blueprints.geoms.Sphere`
	class specifies the type only implicitly in its name. To retrieve the correct class in
	reconstruction from xml then, those implicit attributes are gathered in the class 
	property :attr:`blueprints.type.ThingType._DERIVED_ATTR` to explicitly derived later. Adding new attributes to it
	is done by setting the class property ``_NEW_DERIVED_ATTR`` of the Things Type as a set of attribute 
	names.

	To ensure readability of the resulting xml, attributes that are set to their mujoco 
	defaults are omitted from the xml representations (even though all attributes are set
	in the Thing instance). To check whether an attributes value equals its default, it can
	be obtained from the ``_DEFAULT_VALS`` class property, which is structured as a 
	dictionary of attribute names as keys and their defaults as values. New default values 
	of inheriting classes are added analogously to the previous cases by setting them in a 
	class property ``_NEW_DEFAULT_VALS`` of its Type.
	
	Parameters
	----------
	name : str | None, optional
		This is the name for the object specified by the user. It differs from other 
		user specified properties, in that it might be altered without the users knowledge
		to resolve name conflicts.
	parent : blue.NodeThingType | None, optional
		The parent to which the Thing is attach, if it unattached parent is None.
	**kwargs
		The aggregation of keyword arguments that have not yet been caught by other inheriting 
		Things `__init__` are not used and serve as a dummy variable.
	
	Attributes
	----------
	ID : int
		Each Thing has a unique ID that is gets on initialization from the global REGISTER.
	"""
	
	@restrict
	def __init__(self, 
		     name:   str|None = None, 
		     parent: blue.NodeThingType|None = None, 
		     **kwargs):
		"""
		Parameters
		----------
		name : str | None, optional
			This is the name for the object specified by the user. It differs from other 
			user specified properties, in that it might be altered without the users knowledge
			to resolve name conflicts.
		parent : blue.NodeThingType | None, optional
			The parent to which the Thing is attach, if it unattached parent is None.
		**kwargs
			The aggregation of keyword arguments that have not yet been caught by other inheriting 
			Things `__init__` are not used and serve as a dummy variable.
		"""
		self.ID           = blue.REGISTER.get_ID()
		self._name_scope  = None
		self._name        = name or f'anonymous_{self.__class__.__name__.lower()}'
		self.parent       = parent


	#def __setattr__(self, attr, value):
		"""
		If the Thing is attach to a :class:`blueprints.World` it is informed that its child 
		was altered such that on the next method call on the World, that require the current 
		build, that it needs to rebuild.
		
		Parameters
		----------
		attr : str
			The name of the attribute to be set.
		value : object
			The value assigned to the attribute.
		"""
		#root = self.root
		#if isinstance(root, blue.WorldType) and not isinstance(self, blue.WorldType):
		#	root.built = False
		#super().__setattr__(attr, value)


	def __str__(self) -> str:
		"""
		Returns
		-------
		str
			The representation of a Thing is the xml tag. Children are not included and not 
			all properties are necessarily represented in the xml tag, either if they are 
			located in an external Thing as in :class:`blueprints.geoms.Mesh` or is they are 
			set to default values in which case they are omitted from xml to ensure readability.
		"""
		if hasattr(self, '_MUJOCO_OBJ'):
			xml_element = xml.Element(self._MUJOCO_OBJ, **self._mujoco_specs())
			return xml.tostring(xml_element, encoding='unicode')
		else:
			return repr(self)


	def __repr__(self) -> str:
		"""
		Returns
		-------
		str
			A Thing is represented starting with its class name followed by its name property and 
			its ID.
		"""
		return f'{self.__class__.__name__}<{self.name}:{self.ID}>'


	@restrict
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
		self._xml_root = xml.SubElement(parent, 
						self._MUJOCO_OBJ, 
						**self._mujoco_specs(kwargs))
		return self._xml_root


	@restrict
	def copy(self, **kwargs) -> blue.ThingType:
		"""
		This method constructs a copy of the Thing with possible alterations to its 
		attributes as specified in kwargs.
		
		Parameters
		----------
		**kwargs
			Keyword arguments are passed to the Thing.__init__ to replace the attributes 
			of the current Thing from which copy is called.
		
		Returns
		-------
		blue.ThingType
			A new instance of the Thing
		"""
		blueprint_specs = self._blueprint_specs()
		blueprint_specs.update(kwargs)
		#print('kw', kwargs, blueprint_specs['pos'])
		if 'name' not in kwargs:
			blueprint_specs['name'] = self._name
		thing = self.__class__(**blueprint_specs)
		return thing


	def _clear_step_cache(self):
		"""
		Clears the cached properties derived from simulation values after each time step.
		"""
		for attr in self._STEP_CACHE():
			self.__delattr__(attr)


	@property
	def _location_range(self) -> tuple[np.ndarray, np.ndarray]|None:
		"""
		This is a dummy function that is used by other Things that have a physical size and 
		location to determine the range of space it occupies.
		
		Returns
		-------
		None | None
			None
		"""
		return None


	@property
	def _launched(self):
		"""
		This property indicates whether the world the Thing belongs to has been build 
		and a simulation is launched.

		Returns
		-------
		bool
		"""
		root = self.root
		return isinstance(root, blue.WorldType) and root._built

	# XML INITIALIZATION METHODS

	@restrict
	@classmethod
	def _xml_element_args(cls, xml_element: xml.Element) -> tuple:
		"""
		This method extracts the attributes from xml elements and converts them into 
		a type compatible to the Things corresponding attribute.
		
		Parameters
		----------
		xml_element : xml.Element
			The xml element from which attributes are reconstructed.
		
		Returns
		-------
		tuple
			Three dictionaries are returned, the `init_args` containing arguments passed 
			to :meth:`Thing.__init__`, the `post_args` containing the arguments that are 
			set via :meth:`Thing.__setattr__` after initialization and `rest_args` 
			containing those arguments which must be handled manually.
		"""
		init_args = dict()
		post_args = dict()
		rest_args = dict()
		for key, val in xml_element.items():
			if key in cls._DERIVED_ATTR():
				arg_type = cls._DERIVED_ATTR()[key]
				rest_args[key] = cls._convert_from_string(val, arg_type)
			elif key in cls._BLUEPRINT_ATTR():
				arg_type = cls._BLUEPRINT_ATTR()[key]
				init_args[key] = cls._convert_from_string(val, arg_type)
			elif key in cls._MUJOCO_ATTR():
				arg_type = cls._MUJOCO_ATTR()[key]
				if not val:
					continue
				post_args[key] = cls._convert_from_string(val, arg_type)
			else:
				rest_args[key] = val
		return init_args, post_args, rest_args


	@restrict
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
			The reconstructed Thing.
		"""
		init_args, post_args, rest_args = cls._xml_element_args(xml_element)
		init_args['copy'] = False
		obj = object.__new__(cls)
		obj.__init__(**init_args)
		for key, val in post_args.items():
			setattr(obj, key, val)
		return obj


	@restrict
	def _mujoco_specs(self, specs: dict|None = None) -> dict:
		"""
		This method assists the construction of the Things xml representations. 
		It constructs a dictionary containing the names of the attributes as keys 
		and the string representations of their values.
		
		Returns
		-------
		dict
			A dictionary from which the xml tags attributes are build.
		
		Parameters
		----------
		specs : dict | None, optional
			Specs can be set as a dictionary to overwrite some attributes with custom 
			values.
		"""
		specs        = specs or {}
		condition    = lambda name, attr: attr is not None and (name not in self._DEFAULT_VALS() or np.any(attr != self._DEFAULT_VALS()[name]))
		#convert      = lambda x: str(x) if not isinstance(x, np.ndarray) else self._numpy_to_string(x)
		#convert      = lambda x: str(x).lower() if not isinstance(x, np.ndarray) else self._numpy_to_string(x)
		convert      = self._convert_to_string
		mujoco_attrs = map(lambda name: (name, self.__getattribute__(name)), self._MUJOCO_ATTR().keys())
		mujoco_specs = {name: convert(attr) for name, attr in mujoco_attrs if condition(name, attr)}
		conv_specs   = {name: convert(attr) for name, attr in specs.items()}
		mujoco_specs.update(conv_specs)
		for spec, val in specs.items():
			if val is None:
				del mujoco_specs[spec]
		return mujoco_specs


	@restrict
	def _blueprint_specs(self, specs: dict|None = None) -> dict:
		"""
		This method is used to assist the copying of a Thing. It obtains all attributes 
		from the Thing and potentially copies them to avoid multiple Things referencing 
		the same attribute value.
		
		Returns
		-------
		dict
			The returned dictionary has the attribute names as keys and their corresponding
			values.
		
		Parameters
		----------
		specs : dict | None, optional
			Specs can be specified as a dictionary to overwrite some attributes with custom 
			values.
		"""
		specs           = specs or {}
		#condition       = lambda attr, name: True
		condition       = lambda name, attr: attr is not None and (name not in self._DEFAULT_VALS() or np.any(attr != self._DEFAULT_VALS()[name]))
		convert         = lambda name, attr: attr if name in self._NO_COPY_ATTR() else copy(attr)
		blueprint_attrs = map(lambda name: (name, self.__getattribute__(name)), self._BLUEPRINT_ATTR().keys())
		#converted_attrs = {name: convert(name, attr) for name, attr in blueprint_attrs if condition(name, attr)}
		blueprint_specs = {name: convert(name, attr) for name, attr in blueprint_attrs if condition(name, attr)}
		#blueprint_specs = {name: attr if name not in self._COPY_BLUE_ATTR else attr.copy() for name, attr in converted_attrs.items()}
		blueprint_specs.update(specs)
		for spec, val in specs.items():
			if val is None:
				del blueprint_specs[spec]
		return blueprint_specs

	# XML CONVERSION METHODS

	@restrict
	@classmethod
	def _convert_to_string(cls, obj) -> str:
		"""
		This method converts the string representation of a value from an xml 
		attribute into its corresponding blueprints attribute.
		
		Parameters
		----------
		string : str
			value representation 
		dtype : TYPE
			data type to which the value is to be converted
		
		Returns
		-------
		object
			The returned value can be assigned to the corresponding Things attribute.
		"""
		if isinstance(obj, bool):
			return cls._bool_to_string(obj)
		elif isinstance(obj, np.ndarray):
			return cls._numpy_to_string(obj)
		else:
			return str(obj)

	@restrict
	@classmethod
	def _convert_from_string(cls, string: str, dtype) -> object:
		"""
		This method converts the string representation of a value from an xml 
		attribute into its corresponding blueprints attribute.
		
		Parameters
		----------
		string : str
			value representation 
		dtype : TYPE
			data type to which the value is to be converted
		
		Returns
		-------
		object
			The returned value can be assigned to the corresponding Things attribute.
		"""
		if dtype == bool:
			return cls._string_to_bool(string)
		elif dtype == np.ndarray:
			return cls._string_to_numpy(string)
		else:
			return dtype(string)


	@restrict
	@staticmethod
	def _string_to_numpy(string: str) -> np.ndarray:
		"""
		Convertes a string representation of a value to np.ndarray.
		
		Parameters
		----------
		string : str
			value representation
		
		Returns
		-------
		np.ndarray
			reconstructed np.ndarray
		"""
		sep = ',' if ',' in string else ' '
		return np.array(list(map(float, string.split(sep))))


	@restrict
	@staticmethod
	def _numpy_to_string(array: np.ndarray) -> str:
		"""
		Convertes a np.ndarray to a string representation of its value.
		
		Parameters
		----------
		array : np.ndarray
			any np.ndarray
		
		Returns
		-------
		str
			reconstructed string representation
		"""
		return str(array) if array.shape == () else ' '.join(map(str, array))


	@restrict
	@staticmethod
	def _string_to_bool(string: str) -> bool|None:
		"""
		Convertes a string representation of a value boolean.
		
		Parameters
		----------
		string : str
			value representation
		
		Returns
		-------
		bool
			reconstructed boolean
		
		Raises
		------
		ValueError
			If the string is not 'false', 'true' in any capitalization an Error is raised
		"""
		clean_string = string.lower().strip()
		if clean_string == 'true':
			return True
		elif clean_string == 'false':
			return False
		elif clean_string == 'auto':
			return None
		else:
			raise ValueError(f"_string_to_bool takes either 'true' or 'false' in any capitalization as argument, got {string}")


	@restrict
	@staticmethod
	def _bool_to_string(value: bool|None) -> str:
		"""
		Converts a boolean to a string representation of its value.
		
		Parameters
		----------
		value : bool
			any boolean value
		
		Returns
		-------
		str
			reconstructed string representation
		"""
		if value is None:
			return 'auto'
		return str(value).lower()

	# KINEMATIC TREE PROPERTIES

	@property
	def root(self) -> blue.ThingType:
		"""
		The root attribute is immutable and changes only if the structure of the kinematic tree is 
		altered.

		Returns
		-------
		blue.ThingType
			Retrieves the root of the kinematic tree the Thing in part of, possibly 
			the Thing itself if it has no parent.
		"""
		root = self
		while root.parent is not None:
			root = root.parent
		return root


	@property
	def path(self) -> list:
		"""
		The path attribute is immutable and changes only if the structure of the kinematic tree is 
		altered.

		Returns
		-------
		list
			Retrieves a list of all Things that lead from the Thing itself to its root.
		"""
		root = self
		path = [root]
		while root.parent is not None:
			path = [root.parent] + path
			root = root.parent
		return path


	@property
	def parent(self) -> blue.ThingType | None:
		"""
		The parent attribute is mutable. However it is recommended to use the :meth:`NodeThing.attach` method to alter 
		the kinematic tree.

		Returns
		-------
		blue.ThingType | None
			If the Thing is attach in a kinematic tree, its parent is returned else None.
		"""
		if hasattr(self, '_parent'):
			return self._parent
		else:
			return None


	@parent.setter
	@restrict
	def parent(self, parent: blue.ThingType|None) -> None:
		"""
		The parent attribute is mutable. However it is recommended to use the :meth:`NodeThing.attach` method to alter 
		the kinematic tree.

		Parameters
		----------
		parent : blue.ThingType | None
			The parent references another Thing in a kinematic tree to which the Thing 
			is attached. Setting this property manually is generally not recommended since 
			failure to also attach it to its new parent may lead to unintended consequences.
			Modifications of the kinematic tree should be done via the :meth:`NodeThing.attach` 
			or the :meth:`NodeThing.detach` methods instead.
		"""
		self._parent = parent

	# MUJOCO PROPERTIES

	@property
	def name(self) -> str:
		"""
		The returned name might differ from the user specification by an enumerations 
		scheme that is applied if two Things of the same type in a kinematic tree share 
		the same name.
		
		Returns
		-------
		str
			possibly extended name of the Thing
		"""
		if self._name_scope is not None:
			return self._name_scope.name(self)
		else:
			return self._name


	@name.setter
	@restrict
	def name(self, name: str):
		"""
		Parameters
		----------
		name : str
			The name assigned to the Thing might be altered by enumeration in the case of 
			a naming conflict.
		"""
		if self._name_scope is not None:
			#self._name_scope.assign(self, name)
			assert not self.root._built
			raise Exception("""Name are only allowed to be changed for Things in unbuild Worlds.
To change the of this Thing use :meth:`World.unbuild`.""")
		else:
			self._name = name

	# INHERITANCE PROPERTIES

	@classmethod
	#@property
	def _NO_COPY_ATTR(cls) -> dict:
		"""
		Returns
		-------
		dict
			Attributes in this dictionary will not be copied when the Thing is copied.
		"""
		if hasattr(cls, '_NEW_NO_COPY_ATTR'):
			NO_COPY_ATTR = cls._NEW_NO_COPY_ATTR.copy()
		else:
			NO_COPY_ATTR = set()
		for base in cls.__bases__:
			if hasattr(base, '_NO_COPY_ATTR'):
				NO_COPY_ATTR.update(base._NO_COPY_ATTR())
		return NO_COPY_ATTR


	@classmethod
	#@property
	def _SINGLE_CHILD_ATTR(cls) -> dict:
		"""
		Returns
		-------
		dict
			Attributes in this dictionary will not be copied when the Thing is copied.
		"""
		if hasattr(cls, '_NEW_SINGLE_CHILD_ATTR'):
			SINGLE_CHILD_ATTR = cls._NEW_SINGLE_CHILD_ATTR.copy()
		else:
			SINGLE_CHILD_ATTR = dict()
		for base in cls.__bases__:
			if hasattr(base, '_SINGLE_CHILD_ATTR'):
				SINGLE_CHILD_ATTR.update(base._SINGLE_CHILD_ATTR())
		return SINGLE_CHILD_ATTR


	@classmethod
	#@property
	def _DERIVED_ATTR(cls) -> dict:
		"""
		Returns
		-------
		dict
			Attributes in this dictionary will have to be derived by the global `REGISTER`.
		"""
		if hasattr(cls, '_NEW_DERIVED_ATTR'):
			DERIVED_ATTR = cls._NEW_DERIVED_ATTR.copy()
		else:
			DERIVED_ATTR = dict()
		for base in cls.__bases__:
			if hasattr(base, '_DERIVED_ATTR'):
				DERIVED_ATTR.update(base._DERIVED_ATTR())
		return DERIVED_ATTR


	@classmethod
	#@property
	def _STEP_CACHE(cls) -> dict:
		"""
		Returns
		-------
		set
			Attributes in this set are cached for each time step and deleted afterwards.
		"""
		if hasattr(cls, '_NEW_STEP_CACHE'):
			STEP_CACHE = cls._NEW_STEP_CACHE.copy()
		else:
			STEP_CACHE = set()
		for base in cls.__bases__:
			if hasattr(base, '_STEP_CACHE'):
				STEP_CACHE.update(base._STEP_CACHE())
		return STEP_CACHE


	@classmethod
	#@property
	def _MUJOCO_ATTR(cls) -> dict:
		"""
		Returns
		-------
		dict
			Attributes from this dictionary are used to construct the xml representation.
		"""
		if hasattr(cls, '_NEW_MUJOCO_ATTR'):
			MUJOCO_ATTR = cls._NEW_MUJOCO_ATTR.copy()
		else:
			MUJOCO_ATTR = dict()
		for base in cls.__bases__:
			if hasattr(base, '_MUJOCO_ATTR'):
				MUJOCO_ATTR.update(base._MUJOCO_ATTR())
		if hasattr(cls, '_DEL_MUJOCO_ATTR'):
			for attr in cls._DEL_MUJOCO_ATTR:
				if attr in MUJOCO_ATTR:
					del MUJOCO_ATTR[attr]
		return MUJOCO_ATTR


	@classmethod
	#@property
	def _BLUEPRINT_ATTR(cls) -> dict:
		"""
		Returns
		-------
		dict
			Attributes from this dictionary are used to copy the Thing.
		"""
		if hasattr(cls, '_NEW_BLUEPRINT_ATTR'):
			BLUEPRINT_ATTR = cls._NEW_BLUEPRINT_ATTR.copy()
		else:
			BLUEPRINT_ATTR = dict()
		for base in cls.__bases__:
			if hasattr(base, '_BLUEPRINT_ATTR'):
				BLUEPRINT_ATTR.update(base._BLUEPRINT_ATTR())
		if hasattr(cls, '_DEL_BLUEPRINT_ATTR'):
			for attr in cls._DEL_BLUEPRINT_ATTR:
				if attr in BLUEPRINT_ATTR:
					del BLUEPRINT_ATTR[attr]
		return BLUEPRINT_ATTR


	@classmethod
	#@property
	def _DEFAULT_VALS(cls) -> dict:
		"""
		Returns
		-------
		dict
			This dictionary stores the default values of all attributes.
		"""
		if hasattr(cls, '_NEW_DEFAULT_VALS'):
			DEFAULT_VALS = cls._NEW_DEFAULT_VALS.copy()
		else:
			DEFAULT_VALS = dict()
		for base in cls.__bases__:
			if hasattr(base, '_DEFAULT_VALS'):
				DEFAULT_VALS.update(base._DEFAULT_VALS())
		return DEFAULT_VALS
