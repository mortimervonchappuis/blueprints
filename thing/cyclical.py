"""
.. note::
	If you found yourself here not being a dev, you should probably :mod:`skip this section <blueprints.thing.moveable>` or 
	look up the correspinging docs page of the Thing you were looking after for which this module only 
	implements its corresponding mirror types. A link to the docs page is to be found in the Types doc 
	string.
"""

import blueprints as blue
from blueprints import restrict
from . import base

import numpy as np
import inspect
import xml.etree.ElementTree as xml
from copy import copy
from collections import defaultdict



class CyclicalThing(blue.CyclicalThingType, base.BaseThing):
	"""
	If a Thing has a reference, that potentially points above its parent in the hierarchy of the 
	kinematic tree it becomes a kinematic graph, loosing important properties that are relied on for 
	copying. This class is used to enable migration of attributes that might induce a cycle and to 
	mitigated the introduction of a cycle. The :class:`CyclicalThing` should only be properly attached 
	to one parent, keeping the other references back into the kinematic tree as separate non children 
	attributes (making them improper parents). If a reference of a parent (proper and improper) is 
	performing a copy it should not assign a copy of this instance to its attribute, but instead use 
	the :meth:`_migrate` method. This is used to ensure that this instance is not copied once for each 
	parent.

	Additionally, attributes are restricted from being set to another Thing, if it is not part of the 
	same kinematic tree. This is necessary to prevent exuberant duplications of :math:`c^p` with :math:`c` as the 
	number of copies and :math:`p` as the number of parents (proper and improper).

	If a modification of the kinematic tree results in a parent reference that points outside the 
	kinematic tree, the reference is reset to None.

	The class definition must define two sets ``_PARENT_REFERENCES`` (proper) and ``_OTHER_REFERENCES`` 
	(improper) as class attributes containing the attribute names under which the parents are assigned 
	to the Thing.
	"""
	def __init__(self, **kwargs):
		"""
		Parameters
		----------
		**kwargs
			Keyword arguments are passed to `super().__init__`
		"""
		assert hasattr(self, '_PARENT_REFERENCE')
		assert hasattr(self, '_OTHER_REFERENCES')
		self._MIGRATIONS    = {}
		self._MIGRATED      = False
		self._IGNORE_CHECKS = False
		super().__init__(**kwargs)


	@property
	def _MIGRATION_DONE(self) -> bool:
		"""
		Returns
		-------
		bool
			Indicates if all references have been migrated.
		"""
		return 'parent' in self._MIGRATIONS.keys() and all(ref in self._MIGRATIONS.keys() for ref in self._OTHER_REFERENCES.keys())


	def _start_migration(self, attr, ref, **kwargs):
		"""
		This method creates a singular copy and assigns all previously gather migrations to it.
		Afterwards all migrations are automatically performed as attribute setting on the copy.
		
		Parameters
		----------
		attr : str
			Name of the attribute.
		ref : ThingType
			The Thing that the attribute is referencing (must be proper parent).
		**kwargs
			Keywords arguments are passed to ```copy`` and will be assigned to the new copy.
		
		Returns
		-------
		ThingType
			The returned value is the copy, possibly not fully constructed, since other improper 
			parents in the kinematic tree might still have to be migrated.
		"""
		self._COPY = self.copy(**kwargs)
		self._COPY._IGNORE_CHECKS = True
		self._MIGRATED = True
		ref.attach(self._COPY, copy=False)
		for attr, ref in self._MIGRATIONS.items():
			self._COPY.__setattr__(attr, ref)
		self._MIGRATIONS[attr] = ref
		self._COPY._IGNORE_CHECKS = False


	def _finalize_migration(self):
		"""
		This method finalizes the migration, after the last reference was assigned.
		"""
		self._COPY._IGNORE_CHECKS = False
		self._COPY       = None
		self._MIGRATED   = False
		self._MIGRATIONS = {}


	def _migrate(self, attr: str, ref: blue.ThingType, ignore_checks=False):
		"""
		This method is used to avoid unintended copies by ensuring, that when this Thing is copied 
		as part of a deep copy of the kinematic tree, only one copy of this instance is generated and 
		all other parents (proper and improper) get assigned just this one copy. Generally parents copy 
		all their children when they are copied, which leads to exuberant duplicate in the case of 
		multiple parents.

		This method is used to differentiate between pre and post copy attribute assignment of 
		proper and improper parents. If an improper parent calls for migration before the proper 
		parent, the improper migration gets saved for later. If the proper parent calls for migration 
		a copy is constructed and all previous migrations are performed. Afterwards all migrations 
		result in an attribute assignment.
		
		Parameters
		----------
		attr : str
			Name of the parent attribute
		ref : blue.ThingType
			The reference to the new parent.
		ignore_checks : bool, optional
			If set the restriction that improper parent attributes are allowed to be assigned to other 
			nodes in the kinematic tree is lifted for the duration of this migration.
		
		Raises
		------
		Exception
			If a reference gets migrated multiple times an error is raised.
		"""
		if attr == 'parent':#in self._PARENT_REFERENCES:
			if self._MIGRATED:
				self._finalize_migration()
			self._start_migration(attr, ref)
		elif self._MIGRATED:
			if attr in self._MIGRATIONS:
				raise Exception(f'The attribute {attr} has already been migrated!')
			self._MIGRATIONS[attr] = ref
			# MIGRATION STEP
			temp = self._COPY._IGNORE_CHECKS
			self._COPY._IGNORE_CHECKS = ignore_checks
			self._COPY.__setattr__(attr, ref)
			self._COPY._IGNORE_CHECKS = temp
		elif attr in self._OTHER_REFERENCES:
			self._MIGRATIONS[attr] = ref
		#else:
		#	if attr == 'parent':#in self._PARENT_REFERENCES:
		#		if self._MIGRATED:
		#			raise Exception(f'CyclicalThings are only allowed to be migrated from a parent once. {repr(self)} was already migrated by its parent {repr(self.parent)}')
		#		self._start_migration(attr, ref)
		#	elif attr in self._OTHER_REFERENCES:
		#		self._MIGRATIONS[attr] = ref
		if self._MIGRATION_DONE:
			self._finalize_migration()


	def _decouple(self):
		"""
		This method ensures that parent attribute pointing outside the kinematic tree, the 
		attribute gets reset to None.
		"""
		for attr in self._OTHER_REFERENCES:
			ref = self.__getattribute__(attr)
			if self._disjunct(self.parent, ref) and ref is not None:
				ref.detach(self)
				self.__setattr__(attr, None)


	@staticmethod
	def _disjunct(thing_a, thing_b):
		"""
		This method checks, whether two things belong to the same kinematic tree.
		
		Parameters
		----------
		thing_a : ThingType
			One thing_a that is checked on whether it shares a root with thing_b.
		thing_b : ThingType
			One thing_b that is checked on whether it shares a root with thing_a.
		
		Returns
		-------
		bool
			Indicates if the two Things share a kinematic tree.
		"""
		return thing_a is None or thing_b is None or thing_a.root is not thing_b.root


	def __setattr__(self, attr, ref) -> None:
		"""
		This method redirects to various other ``__setattr__`` implementations.
		
		Parameters
		----------
		attr : str
			Name of the attribute
		ref : object
			Value of the attribute
		
		Raises
		------
		ValueError
			If the attribute cannot be set an error is raised.
		"""
		if attr == 'parent':
			if self.parent is not None:
				self.parent.detach(self)
			super().__setattr__('parent', ref)
			if not self._IGNORE_CHECKS:
				self._decouple()
		elif attr in self._OTHER_REFERENCES:
			if self._IGNORE_CHECKS or not self._disjunct(self.parent, ref) or ref is None:
				getattr(self.__class__, attr).fset(self, ref)
			else:
				raise ValueError(f'Setting {attr} is only allowed for references that share the same root.')
		else:
			super().__setattr__(attr, ref)
