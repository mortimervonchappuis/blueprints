"""
.. note::
	If you found yourself here not being a dev, you should probably :mod:`skip this section <blueprints.utils>` from or 
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



class UniqueThing(blue.UniqueThingType, base.BaseThing):

	"""
	Things that have multiple parents and that should not be copied to often can inherit from 
	:class:`UniqueThing`. If the attributes of this Thing should be altered for one parent but stay 
	the same for other parents, a copy is created which then can be altered. If the the Thing has only 
	one parent the attributes are instead altered directly. To do this, parents can call the 
	:meth:`_prepare_for_modification` method passing themselves which returns either a copy of the 
	Thing or the Thing itself which can then be altered.

	>>> alterable_child = node_thing.unique_thing._prepare_for_modification(node_thing)
	>>> atterable_child.attr = value

	This usecase is important if a Thing is modified often, but contains large amounts of data like 
	:class:`blueprints.cache.MeshCache` to avoid redundant copies.
	
	Parameters
	----------
	**kwargs
		Keyword arguments are passed to ``super().__init__``
		
	Attributes
	----------
	freeze : bool
		Setting this attribute freezes the setting of other attributes until it is unset.
	"""

	def __init__(self, **kwargs):
		"""
		Parameters
		----------
		**kwargs
			Keyword arguments are passed to ``super().__init__``
		"""
		self._references = set()
		self.freeze      = False
		super().__init__(**kwargs)
		assert hasattr(self, '_REFERENCE_NAME')


	def __setattr__(self, attr, val):
		"""
		Parameters
		----------
		attr : str
			The name of the attribute
		val : object
			The value assigned to the attribute
		"""
		if attr == 'freeze' or not hasattr(self, 'freeze') or not self.freeze:
			super().__setattr__(attr, val)


	def _remove(self, thing):
		"""
		Parameters
		----------
		thing : ThingType
			thing is detached as a parent of the :class:`UniqueThing`.
		"""
		if thing in self:
			self._references.remove(thing)


	def _add(self, thing):
		"""
		Parameters
		----------
		thing : ThingType
			thing is added as a parent of the :class:`UniqueThing`.
		"""
		if thing not in self:
			if hasattr(thing, self._REFERENCE_NAME):
				getattr(thing, self._REFERENCE_NAME)._remove(thing)
			self._references.add(thing)
			setattr(thing, self._REFERENCE_NAME, self)


	def _prepare_for_modification(self, parent):
		"""
		Parameters
		----------
		parent : ThingType
			The parent is given such that if a copy is generated if can be detached from the 
			:class:`UniqueThing` parents.
		"""
		assert getattr(parent, self._REFERENCE_NAME) is self
		assert parent in self
		#if self.parent is not None:
		#	self.root = self.parent.root
		if len(self._references) > 1 and not self.freeze:# and (not isinstance(self.root, blue.WorldType) or not self.root._is_built):
			thing = self.copy()
			thing._add(parent)


	def __contains__(self, thing):
		"""
		Parameters
		----------
		thing : ThingType
			A potential parent of this instance
		
		Returns
		-------
		bool
			Indicates whether thing is a parent of this instance.
		"""
		return thing in self._references
