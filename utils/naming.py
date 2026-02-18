from collections import defaultdict



class NameManager:

	"""
	The Namemanager is a attribute of the root Thing in a kinematic tree. It resolves name conflicts
	by finding synonymous Things and renaming them by adding a numeration. If a renamin results in a
	new naming conflicts, another round of renaming is applied unti all conflicts are resolved.
	
	For example, if there are four Geoms in a kinematic tree that are named by the user respectively:
	
	.. code-block::
		:caption: USERNAMES
				
		"rock", "rock", "rock_(1)", "stone"
	
	A name conflict exists for the first two geoms. The name conflict resolving after the first round 
	would result in the names

	.. code-block::
		:caption: RENAMING (1)

		"rock_(0)", "rock_(1)", "rock_(0)", "stone"
	
	Now a name conflict for the first and the third name exists, because the user name for the third 
	Geom coincidentally alligns with the numeration scheme of the resolver. Therefore a second renaming 
	round results in the names

	.. code-block::
		:caption: RENAMING (2)

		"rock_(0)_(0)", "rock_(1)", "rock_(0)_(1)", "stone"
	
	The cascading of numerations can be avoided by not ending user defined names with ``"_(<int>)"``.
	
	Attributes
	----------
	root : ThingType
		The root node for which the NameManager resolves nameing conflicts.
	type_scopes : dict
		Identical names only result in conflicts if they are given to Thing of the same type. The type_scopes dict splits the name resolving for all objects of one type.
	"""
	
	def __init__(self, root):
		"""
		The NameManager resolves naming conflicts for predefined names from the user and naming conflicts resulting from renaming of specific objects.
		
		Parameters
		----------
		root : ThingType
			A thing which is the root if its kinematic tree.
		"""
		self.root = root
		self.type_scopes = {}
		self.register()


	def __getitem__(self, key):
		"""
		Returns the type scope for a given Type.
		
		Parameters
		----------
		key : class
			The class for which the TypeScope is requested.
		
		Returns
		-------
		TypeScope
			The TypeScope that handles the name conflict resolving for all descendants of root of the given type.
		"""
		return self.type_scopes[key]


	def __setitem__(self, key, type_scope):
		"""
		Sets the TypeScope for the a given key Type.
		
		Parameters
		----------
		key : class
			The Type of objects for which the TypeScope is set.
		type_scope : TypeScope
			The TypeScope assigned to the NameManager.
		"""
		self.type_scopes[key] = type_scope


	def register(self):
		"""
		The calling of this method resolves naming conflicts.
		"""
		DESCENDANTS = defaultdict(list)
		for _, dec_dict in self.root.descendants.items():
			element_type = dec_dict['type']
			descendants  = dec_dict['descendants']
			DESCENDANTS[element_type].extend(descendants)
		#for name, dec_dict in self.root.descendants.items():
		for element_type, descendants in DESCENDANTS.items():
			descendants = [descendant for i, descendant in enumerate(descendants) if descendants.index(descendant) == i]
			#element_type = dec_dict['type']
			#descendants  = dec_dict['descendants']
			self[element_type] = TypeScope(self, descendants)


	def unregister(self):
		"""
		The calling of this method reverses the effect of registering the NameManager.
		"""
		for type_scope in self.type_scopes.values():
			type_scope.unregister()
		self.type_scopes = {}


class TypeScope:

	"""
	This class upon initialization resolves name conflicts for all elements of a specifig Type.
	After renaming of assigned Things by the user, the `register` method is called automatically 
	to resolve potentially new conflicts.
	
	Attributes
	----------
	descendants : list
		A list of all Things that are captured under the TypeScope.
	name_manager : NameManager
		The NameManager to which the TypeScope is assigned.
	name_scopes : dict
		A dictionary containing the NameScopes for each user defined name.
	"""
	
	def __init__(self, name_manager, descendants):
		"""
		Upon initialization name conflicts are resolved for all descendants.
		
		Parameters
		----------
		name_manager : NameManager
			The NameManager to which the TypeScope is assigned.
		descendants : list
			A list of all Things that are captured under the TypeScope.
		"""
		self.name_manager = name_manager
		self.descendants  = descendants
		self.name_scopes  = {}
		self.register()


	def __getitem__(self, name: str):
		"""
		Returns the NameScope for a given name.
		
		Parameters
		----------
		name : str
			The name of the requested NameScope
		
		Returns
		-------
		NameScope
			The requested NameScope contains all Things decending from root, that are of the same Type and share a userdefined name.
		"""
		return self.name_scopes[name]


	def __setitem__(self, name: str, name_scope) -> None:
		"""
		Assigns a NameScope for a given name.
		
		Parameters
		----------
		name : str
			The name for which the NameScope is assigned.
		name_scope : NameScope
			The NameScope which is assigned to the given name.
		"""
		self.name_scopes[name] = name_scope


	def register(self):
		"""
		Calling this method resolves name conflicts for all assigned descendants of root. 
		After a assigned Thing is renamed register is called automatically to resolve 
		potential new conflicts.
		"""
		for descendant in self.descendants:
			descendant._name_scope = None
		self.name_scopes  = {name: NameScope(type_scope=self, 
						     name=name, 
						     descendants=descendants) for name, descendants in self.synonyms().items()}
		while True:
			synonyms = self.synonyms()
			if all(map(lambda x: len(x) == 1, synonyms.values())):
				break
			else:
				for name, descendants in synonyms.items():
					if len(descendants) > 1:
						self[name] = NameScope(self, name, descendants)


	def unregister(self):
		"""
		This method reverses the effect of registering for the TypeScope.
		"""
		for name_scope in self.name_scopes.values():
			name_scope.unregister()
		self.name_scopes = {}
		self.decendants  = {}


	def synonyms(self):
		"""
		This method gathers Things that are synomymous (share the same name). 
		This method takes renaming into acount, so after one round of renaming 
		has taken place, calling this method again, will result in synonyms 
		according to the previous renaming.
		
		Returns
		-------
		dict
			Keys of the dictionary are names and the values are lists containing all Things under the scope of the given name.
		"""
		synonyms = defaultdict(list)
		for descendant in self.descendants:
			synonyms[descendant.name].append(descendant)
		return synonyms



class NameScope:

	"""
	The NameScope class resolves the name conflicts for Things of the same Type 
	and the same assigned name (either by the user or by a previous name conflict 
	resolving iteration).
	
	Attributes
	----------
	descendants : list
	A list of all descendants of the same Type with the same name.
	type_scope : TypeScope
	The TypeScope to which the NameScope is assigned.
	"""
	
	def __init__(self, type_scope, name, descendants):
		"""
		After the initialization the name conflicts for all given devendents are resolved.
		
		Parameters
		----------
		type_scope : TypeScope
			The TypeScope to which the NameScope is assigned.
		name : str
			The initial name of all Things assigned to the NameScope.
		descendants : list
			A list of all descendants of the same Type with the same name.
		"""
		self.descendants = descendants
		self._name       = name
		self.type_scope  = type_scope
		self.register()


	def __len__(self):
		"""
		Gives the number of Things under this scope.
		
		Returns
		-------
		int
			The amount of assigned descendants.
		"""
		return len(self.descendants)


	def register(self):
		"""
		This method assignes all descendants to this NameScope. Afterwards 
		a unique name can be requested by the descendant.
		"""
		for descendant in self.descendants:
			descendant._name_scope = self


	def unregister(self):
		"""
		This method reverses the effect of registering all decendents of 
		the NameScope.
		"""
		for descendant in self.descendants:
			descendant._name_scope = None
		self.decendants = {}


	def name(self, descendant) -> str:
		"""
		Returns a unique name for a given descendant.
		
		Parameters
		----------
		descendant : Thing
			The Thing that is assigned to the NameScope.
		
		Returns
		-------
		str
			A unique name that is potentially altered by an enumeration scheme to resolve name conflicts.
		"""
		if len(self) == 1:
			return self._name
		else:
			idx = self.descendants.index(descendant)
			return f'{self._name}_({idx})'


	def __assign(self, descendant, name):
		"""
		This method assigns a name to a descendant.
		
		Parameters
		----------
		descendant : Thing
			A descendant to which a name is assigned.
		name : str
			The name to be assigned to the descendant.
		"""
		if name != self._name and name != descendant.name:
			descendant._name = name
			self.type_scope.register()
