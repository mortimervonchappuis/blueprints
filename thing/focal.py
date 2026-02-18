import blueprints as blue
from blueprints import restrict
from . import base

import numpy as np
import inspect
import xml.etree.ElementTree as xml



class FocalThing(blue.FocalThingType, base.BaseThing):
	"""
	This class is similar in behavior to :class:`CyclicalThing <blueprints.thing.cyclical.CyclicalThing>` 
	with the exception that it does not have a concrete parent. 
	FocalThings are the target of multiple incomming attribute 
	assignments from higher up in the hierarchy hence merging 
	different path and introducing cycles. To deal with copies 
	of kinematic graphs containing FocalThings, migration methods 
	similar to those in :class:`CyclicalThing <blueprints.thing.cyclical.CyclicalThing>` are defined.
	
	.. note::
		This Type is used in subclassing checks to summarize multile concrete ThingTypes.
	"""
	
	@restrict
	def __init__(self, *args, **kwargs) -> None:
		super().__init__(*args, **kwargs)
