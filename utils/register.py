import blueprints as blue



class Register:

	"""
	The Register is used to assign Thing IDs and to amalgamate all Thing classes for object recostruction from an xml element.
	
	Attributes
	----------
	ACTUATOR_THINGS : dict
		A dictionary containing the respective Actuator classes with their xml tags as keys.
	ASSET_THINGS : dict
		A dictionary containing the respective Asset classes with their xml type attributes as keys.
	assets : list
		A list of all assets that have been created.
	BASIC_THINGS : dict
		A dictionary containing basic Thing classes with their xml tags as keys.
	CACHE_THINGS : dict
		A dictionary containing the respective Cache classes with their xml type attributes as keys.
	caches : list
		A list of all caches that have been created
	DERIVED_THINGS : dict
		A dictionary containing dictionaries of Things for which the specific class is derived from the xml type attribute.
	GEOM_THINGS : dict
		A dictionary containing the respective Geom classes with their xml type attributes as keys.
	JOINT_THINGS : dict
		A dictionary containing the respective Joint classes with their xml type attributes as keys.
	SENSOR_THINGS : dict
		A dictionary containing the respective Sensor classes with their xml tags as keys.
	SITE_THINGS : dict
		A dictionary containing the respective Actuator classes with their xml type attributes as keys.
	"""
	
	BASIC_THINGS    = {'body':   blue.Body, 
			   'camera': blue.Camera, 
			   'light':  blue.Light}
	GEOM_THINGS     = blue.geoms.GEOM_THINGS
	SITE_THINGS     = blue.sites.SITE_THINGS
	JOINT_THINGS    = blue.joints.JOINT_THINGS
	ASSET_THINGS    = blue.assets.ASSET_THINGS
	SENSOR_THINGS   = blue.sensors.SENSOR_THINGS
	ACTUATOR_THINGS = blue.actuators.ACTUATOR_THINGS
	CACHE_THINGS    = blue.cache.CACHE_THINGS
	DERIVED_THINGS  = {'geom':  GEOM_THINGS, 
			   'site':  SITE_THINGS, 
			   'joint': JOINT_THINGS}

	def __init__(self):
		"""
		The Register is used twofold, to keep track of objects and to assign blueprint Thing classes to xml tags.
		"""
		self.__ID = 0
		self.assets = []
		self.caches = []
		self.copy_root = None


	def get_ID(self):
		"""
		This method creates a new ID on each call that can be assigned to a newly initialized Thing. The internal ID counter is incremented afterwards.
		
		Returns
		-------
		int
			A unique ID for a freshly initialized object.
		"""
		ID = self.__ID
		self.__ID += 1
		return ID


	@classmethod
	def _get_thing_class(cls, 
			     xml_element):
		"""
		This method finds the Thing class for a given xml_element.
		
		Parameters
		----------
		xml_element : TYPE
			Description
		
		Returns
		-------
		TYPE
			Description
		"""
		element_tag = xml_element.tag
		if element_tag in cls.BASIC_THINGS:
			obj_class = cls.BASIC_THINGS[element_tag]
		elif element_tag in cls.DERIVED_THINGS:
			DERIVED_CLASSES = cls.DERIVED_THINGS[element_tag]
			element_type = xml_element.get('type')
			if element_type in DERIVED_CLASSES:
				obj_class = DERIVED_CLASSES[element_type]
		elif element_tag in cls.ASSET_THINGS:
			obj_class = cls.ASSET_THINGS[element_tag]
		elif element_tag in cls.SENSOR_THINGS:
			obj_class = cls.SENSOR_THINGS[element_tag]
		elif element_tag in cls.ACTUATOR_THINGS:
			obj_class = cls.ACTUATOR_THINGS[element_tag]
		return obj_class


	@classmethod
	def _get_thing(cls, 
		       xml_element):
		"""
		This method constructs a Thing from an xml element.
		
		Parameters
		----------
		xml_element : xml.etree.ElementTree.Element
			An element from the xml tree paresed with xml.etree.ElementTree.
		
		Returns
		-------
		ThingType
			A thing constructed from the xml_element.
		"""
		thing_class = cls._get_thing_class(xml_element)
		return thing_class._from_xml_element(xml_element)
