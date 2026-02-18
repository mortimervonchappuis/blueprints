import os
import sys
import struct
import xml.etree.ElementTree as xml
import numpy as np
import blueprints as blue
from collections import defaultdict
from itertools import count
from imageio import imread



class BaseCache(blue.UniqueThing, blue.CacheType):

	"""
	:class:`Caches <BaseCache>` are used to store large amounts of data. For blueprints to allow for the 
	modification of this data, it has to be loaded to memory. If this data is part of a Thing which is copied 
	multiple times, this would drastically increase the memory usage for redundant data. Caches solve this, 
	by implementing a single source of data for multiple copies of a Thing. Thing classes that contain large 
	amounts of data can simply outsource them to a cache (by linking the outsourced properties to the caches 
	properties) while keeping small attributes that are modified frequently. To ensure data integrity, if a 
	Things property which is stored in its :class:`Cache <BaseCache>` is modified, a new copy of the caches 
	content is created replacing the old ``cache`` attribute of the Thing (unless only one Thing uses the 
	Cache, in which case the Cache is modified directly). For example, consider the following scenario:

	.. code-block:: python
		:caption: Cache example — The large data attribute is outsourced to a Cache

		>>> one = SomeThing(data=open('large.file').read())
		>>> one.cache
		SomeCache<None:2>
		>>> one.data is one.cache.data
		True
		>>> two = one.copy()
		>>> one.cache is two.cache
		True
		>>> one.name = 'one'
		>>> two.name = 'two'
		>>> one.cache is two.cache
		True
		>>> two.data = open('other.file').read()
		>>> one.cache is two.cache
		False
	"""

	def __init__(self, **kwargs):
		"""
		Parameters
		----------
		**kwargs
			Keyword arguments are passed to ``super().__init__``.
		"""
		self._DEPENDENCY_FLAGS = defaultdict(lambda: True)
		super().__init__(**kwargs)


	def _flag_dependencies(self, attr):
		"""
		This method updates the flags for deriving attributes, notifying them that their value mus be 
		updated.
		
		Parameters
		----------
		attr : str
			 name of the attribute
		"""
		if hasattr(self, '_DEPENDENCIES') and hasattr(self, '_DEPENDENCY_FLAGS'):
			for dependency in self._DEPENDENCIES[attr]:
				self._DEPENDENCY_FLAGS[dependency] = False


	def _validate(self, attr):
		"""
		Indicates whether the attribute has to be updated.
		
		Parameters
		----------
		attr : str
			 name of the attribute
		"""
		return not hasattr(self, '_DEPENDENCY_FLAGS') or self._DEPENDENCY_FLAGS[attr]



class MeshCache(blue.MeshCacheType, BaseCache):

	"""
	This class stores the mesh data which might be loaded from a file and is too large to be copied frequently.
	
	Attributes
	----------
	built : bool
		A flag indicating whether the MeshAssets has been built. This is necessary since multiple 
		:class:`MeshAssets <blueprints.assets.MeshAsset>` might use the same :class:`MeshCache` in 
		which case it should be built only once.
	centered : bool
		If ``True`` the vertecies are normalized such that their mean position is the reference frames 
		origin.
	vertecies : list
		The list of vertecies. Each vertex is a np.ndarray with 3 components for each of the spatial 
		dimensions.
	texcoords : list
		The list of texture coordinates.
	texcoords_idx : dict
		A dictionary used to link texture coordinates with face vertecies.
	normals : list
		The normals used by mujoco, this attribute is derived and should not be used for modification, 
		see :attr:`face_normals` or :attr:`vertex_normals` for this instead.
	face_normals : list
		Face normals are the normal vectors of faces. They specify the direction in which the face 
		points.
	vertex_normals : list
		Vertex normals are the normal vectors of vertecies. They specify the direction in which the 
		vertecies of a face point. Specifying vertex normals instead of face normals enables mujoco to
		renderer soft edges. 
	normals_idx : TYPE
		A dictionary used to link normals with faces.
	faces : list
		This attribute is used to write to xml. It returns a flattened list of indecies that is 
		written raw to xml. For a structured list of faces see :attr:`faces`.
	filename : str
		The user specified file name.
	"""
	
	@blue.restrict
	def __init__(self, 
		     vertecies: np.ndarray|list[np.ndarray|list[float|int]]|None = None,  
		     faces:     list[np.ndarray|list[float|int]]|None            = None,
		     filename:  str|None = None, 
		     centered:  bool     = False, 
		     **kwargs) -> None:
		"""
		Parameters
		----------
		vertecies : np.ndarray | list[np.ndarray | list[float | int]] | None, optional
			A list of all vertex positions.
		faces : list[np.ndarray | list[float | int]] | None, optional
			A list of all faces. A face consists of at least 3 indecies of vertecies.
		file : str | None, optional
			The filename from which the mesh data is loaded and to to which the mesh will be saved. 
			The file is saved in a special directory such that no input file is overwritten by the 
			output file.
		centered : bool, optional
			If ``True`` the vertecies are normalized such that their mean position is the reference 
			frames origin.
		**kwargs
			Keyword arguments are passed to ``super().__init__``.
		"""
		self.centered = centered
		self.filename = filename
		self.texcoords      = None
		self.texcoords_idx  = None
		self.normals        = None
		self.normals_idx    = None
		self.face_normals   = None
		self.vertex_normals = None
		self._DEPENDENCIES  = {'vertecies': ('vertecies_minimum', 
						     'vertecies_center', 
						     'vertecies_maximum')}
		super().__init__(**kwargs)
		if filename is not None:
			self.load(filename)
		elif vertecies is not None:
			self.vertecies = vertecies
			self.faces     = faces
		self._built = False
		if centered and hasattr(self, '_vertecies'):
			self.vertecies = self.vertecies - self.vertecies_center
		#blue.REGISTER.caches.append(self)
		if 'name' not in kwargs:
			self._name = None


	@blue.restrict
	def copy(self, **kwargs) -> blue.CacheType:
		"""
		This method constructs a copy of the Cache with possible alterations to its 
		attributes as specified in kwargs.
		
		Parameters
		----------
		**kwargs
			Keyword arguments are passed to the :meth:`__init__` to replace the attributes 
			of the current Thing from which copy is called.
		
		Returns
		-------
		blue.CacheType
			A new instance of the Cache
		"""
		mesh = super().copy(**kwargs)
		if 'name' not in kwargs:
			mesh._name = self._name
		return mesh


	@blue.restrict
	def _build(self, dirname: str, **kwargs) -> None:
		"""
		This method is called by the Assets parent to construct the xml representations of 
		the kinematic tree.
		
		Parameters
		----------
		dirname : str
			The directory name in which the mesh file is saved.
		**kwargs
			Dummy argument
		"""
		if not self._built:
			pathname, basename = os.path.split(self.filename)
			if not pathname.endswith(dirname):
				path = f'{dirname}/{self.ID}_{basename}'
			else:
				path = self.filename
			self.save(path)
			self._path = path
			self._built = True


	@blue.restrict
	@classmethod
	def _from_xml_element(cls, xml_element) -> blue.CacheType:
		"""
		This method reconstructs an MeshCache from an xml element.
		
		Parameters
		----------
		xml_element : xml.Element
			The xml element from which a MeshCache is reconstructed.
		
		Returns
		-------
		blue.CacheType
			The reconstructed MeshCache.
		"""
		init_args, post_args, rest_args = cls._xml_element_args(xml_element)
		if 'vertex' in post_args:
			init_args['vertecies'] = post_args['vertex']
			del post_args['vertex']
		obj = object.__new__(cls)
		obj.__init__(**init_args)
		for key, val in post_args.items():
			setattr(obj, key, val)
		return obj


	@blue.restrict
	def load(self, filename: str) -> None:
		"""
		This methods loads the mesh data from a file.
		
		Parameters
		----------
		filename : str
			Possible file types are ``'.stl'`` binary or ascii and ``'.obj'`` ascii.
		"""
		with open(filename, 'rb') as file:
			data = file.read()
		if data.isascii():
			self._load_ascii(filename, data.decode('ascii'))
		else:
			self._load_binary(filename, data)


	@blue.restrict
	def _load_binary(self, 
			 filename: str, 
			 data: bytes) -> None:
		"""
		Helper function that routes the loading for difference file formats in binary.
		
		Parameters
		----------
		filename : str
			Possible file types are ``'.stl'``.
		data : bytes
			The data of the file to be parsed
		
		Raises
		------
		NotImplemented
			If the file format is not supported an error is raised.
		"""
		if filename.lower().endswith('.obj'):
			raise NotImplemented
		elif filename.lower().endswith('.stl'):
			self._load_STL_binary(data)
		else:
			raise NotImplemented

	@blue.restrict
	def _load_ascii(self, 
			filename: str, 
			data: str) -> None:
		"""
		Helper function that routes the loading for difference file formats in ascii.
		
		Parameters
		----------
		filename : str
			Possible file types are ``'.stl'`` and ``'.obj'``.
		data : str
			The data of the file to be parsed
		
		Raises
		------
		NotImplemented
			If the file format is not supported an error is raised.
		"""
		#return
		if filename.lower().endswith('.obj'):
			self._load_OBJ_ascii(data)
		elif filename.lower().endswith('.stl'):
			self._load_STL_ascii(data)
		else:
			raise NotImplemented


	@blue.restrict
	def _load_OBJ_ascii(self, data: str) -> None:
		"""
		Parses the data from an ascii obj file to the Cache.
		
		Parameters
		----------
		data : str
			The data of the file to be parsed
		"""
		# DATA CONTAINERS
		vertecies      = []
		faces          = []
		texcoords      = []
		face_normals   = []
		vertex_normals = []
		# SORTING CONTAINERS
		all_face_normals = {}
		texcoords_idx    = {}
		normals_idx      = {}
		# READING LINES
		for line in data.split('\n'):
			# REMOVING COMMENTS
			line = line.split('#')[0]
			# HANDLING VERTECIES
			if line.startswith('v '):
				values = line[2:].strip().split(' ')[:3]
				vertex = list(map(float, values))
				vertecies.append(np.array(vertex))
			elif line.startswith('vn '):
				values = line[3:].strip().split(' ')[:3]
				normal = list(map(float, values))
				normal = np.array(normal)
				#normal = normal / np.linalg.norm(normal)
				vertex_normals.append(normal)
			elif line.startswith('vt '):
				values   = line[3:].strip().split(' ')[:2]
				texcoord = list(map(float, values))
				texcoords.append(np.array(texcoord))
			elif line.startswith('f '):
				vertex_idx   = line[2:].strip().split(' ')#[:3]
				triangle_idx = [[vertex_idx[0], a, b] for a, b in zip(vertex_idx[1:], vertex_idx[2:])]
				for triangle in triangle_idx:
					if all(map(lambda x: x.count('//') == 1, triangle)):
						# GET VALUES
						values, normal_idx = zip(*(value.split('//') for value in triangle))
						values     = list(map(lambda x: int(x) - 1, values))
						normal_idx = list(map(lambda x: int(x) - 1, normal_idx))
						# SET INDECIES
						normals_idx[len(faces)] = normal_idx
					elif all(map(lambda x: x.count('/') == 2, triangle)):
						# GET VALUES
						values, tex_idx, normal_idx = zip(*(value.split('/') for value in triangle))
						values     = list(map(lambda x: int(x) - 1, values))
						tex_idx    = list(map(lambda x: int(x) - 1, tex_idx))
						normal_idx = list(map(lambda x: int(x) - 1, normal_idx))
						# SET INDECIES
						normals_idx[len(faces)] = normal_idx
						texcoords_idx[len(faces)]  = tex_idx
					elif all(map(lambda x: x.count('/') == 1, triangle)):
						# GET VALUES
						values, tex_idx = zip(*(value.split('/') for value in triangle))
						values  = list(map(lambda x: int(x) - 1, values))
						tex_idx = list(map(lambda x: int(x) - 1, tex_idx))
						# SET INDECIES
						texcoords_idx[len(faces)] = tex_idx
					face = list(map(int, values))
					face = [idx if idx != -2 else len(vertecies) for idx in face]
					faces.append(np.array(face)) # -1
		assert not texcoords_idx.values() or all(map(texcoords_idx.__contains__, range(len(faces))))
		self.vertecies      = vertecies
		self.faces          = faces
		self.vertex_normals = vertex_normals or None
		self.texcoords      = texcoords      or None
		self.texcoords_idx  = texcoords_idx  or None
		self.normals_idx    = normals_idx    or None


	@blue.restrict
	def _load_STL_ascii(self, data: str) -> None:
		"""
		Parses the data from an ascii stl file to the Cache.
		
		Parameters
		----------
		data : str
			The data of the file to be parsed
		"""
		# CROP DATA
		data = data[data.find('\n'):].split('facet')[1::2]
		# INIT CONTAINERS
		faces          = list()
		normals        = list()
		vertecies      = list()
		vertex_counter = count()
		vertex_table   = defaultdict(lambda: next(vertex_counter))
		# VECTOR CONSTRUCTION FUNCTION
		vector_tuple = lambda s: tuple(map(float, list(filter(lambda x: x != '', s.split(' ')))[:3])) 
		for facet in data:
			# GET FACET SECTION
			lines = facet.split('\n')
			# GET VECTORS
			normal_line    = lines[0].replace('normal', '').strip()
			vertex_0_line  = lines[2].replace('vertex', '').strip()
			vertex_1_line  = lines[3].replace('vertex', '').strip()
			vertex_2_line  = lines[4].replace('vertex', '').strip()
			# CONSTRUCT VECTOR TUPLES
			normal_tuple   = vector_tuple(normal_line)
			vertex_tuple_0 = vector_tuple(vertex_0_line)
			vertex_tuple_1 = vector_tuple(vertex_1_line)
			vertex_tuple_2 = vector_tuple(vertex_2_line)
			# CONSTRUCT NP.ARRAYS
			normal_array   = np.array(normal_tuple)
			vertex_array_0 = np.array(vertex_tuple_0)
			vertex_array_1 = np.array(vertex_tuple_1)
			vertex_array_2 = np.array(vertex_tuple_2)
			# ADD NEW VERTECIES TO HASH TABLE
			if vertex_tuple_0 not in vertex_table:
				vertex_table[vertex_tuple_0]
				vertecies.append(vertex_array_0)
			if vertex_tuple_1 not in vertex_table:
				vertex_table[vertex_tuple_1]
				vertecies.append(vertex_array_1)
			if vertex_tuple_2 not in vertex_table:
				vertex_table[vertex_tuple_2]
				vertecies.append(vertex_array_2)
			# GET NORMAL FROM ORDER
			edge_array_0 = vertex_array_1 - vertex_array_0
			edge_array_1 = vertex_array_2 - vertex_array_0
			edge_cross   = np.cross(edge_array_0, edge_array_1)
			edge_normal  = edge_cross/np.linalg.norm(edge_cross)
			correlation = np.dot(edge_normal, normal_array)
			if correlation > 0:
				vertex_tuples = [vertex_tuple_0, vertex_tuple_1, vertex_tuple_2]
				face = list(map(vertex_table.__getitem__, vertex_tuples))
				faces.append(face)
				normals.append(normal_array)
			elif correlation < 0:
				vertex_tuples = [vertex_tuple_0, vertex_tuple_2, vertex_tuple_1]
				face = list(map(vertex_table.__getitem__, vertex_tuples))
				faces.append(face)
				normals.append(-normal_array)
			else:
				vertex_tuples = [vertex_tuple_0, vertex_tuple_2, vertex_tuple_1]
				face = list(map(vertex_table.__getitem__, vertex_tuples))
				faces.append(face)
				normals.append(normal_array)
				vertex_tuples = [vertex_tuple_0, vertex_tuple_1, vertex_tuple_2]
				face = list(map(vertex_table.__getitem__, vertex_tuples))
				faces.append(face)
				normals.append(-normal_array)
		# SET ATTRIBUTES
		self.vertecies    = vertecies
		self.faces        = faces
		self.face_normals = normals


	@blue.restrict
	def _load_STL_binary(self, data: bytes) -> None:
		"""
		Parses the data from an binary stl file to the Cache.
		
		Parameters
		----------
		data : bytes
			The data of the file to be parsed
		"""
		# CONVERSION FUNCTIONS
		bytes_to_float = lambda x: struct.unpack('f', x)[0]
		vector_tuple   = lambda x: tuple(map(bytes_to_float, (x[:4], x[4:8], x[8:])))
		# HEADER DATA
		header         = data[:80]
		number         = struct.unpack('I', data[80:84])[0]
		# TRTIANGLE DATA
		triangle_data  = data[84:]
		faces          = list()
		normals        = list()
		vertecies      = list()
		vertex_counter = count()
		vertex_table   = defaultdict(lambda: next(vertex_counter))
		for i in range(number):
			# GET TRIANGLE
			triangle = triangle_data[i*50:(i+1)*50]
			# CONSTRUCT VECTOR TUPLES
			normal_tuple   = vector_tuple(triangle[:12])
			vertex_tuple_0 = vector_tuple(triangle[12:24])
			vertex_tuple_1 = vector_tuple(triangle[24:36])
			vertex_tuple_2 = vector_tuple(triangle[36:48])
			# CONSTRUCT NP.ARRAYS
			normal_array   = np.array(normal_tuple)
			vertex_array_0 = np.array(vertex_tuple_0)
			vertex_array_1 = np.array(vertex_tuple_1)
			vertex_array_2 = np.array(vertex_tuple_2)
			# ADD NEW VERTECIES TO HASH TABLE
			if vertex_tuple_0 not in vertex_table:
				vertex_table[vertex_tuple_0]
				vertecies.append(vertex_array_0)
			if vertex_tuple_1 not in vertex_table:
				vertex_table[vertex_tuple_1]
				vertecies.append(vertex_array_1)
			if vertex_tuple_2 not in vertex_table:
				vertex_table[vertex_tuple_2]
				vertecies.append(vertex_array_2)
			# GET NORMAL FROM ORDER
			edge_array_0 = vertex_array_1 - vertex_array_0
			edge_array_1 = vertex_array_2 - vertex_array_0
			edge_cross   = np.cross(edge_array_0, edge_array_1)
			edge_normal  = edge_cross/np.linalg.norm(edge_cross)
			correlation = np.dot(edge_normal, normal_array)
			if correlation > 0:
				vertex_tuples = [vertex_tuple_0, vertex_tuple_1, vertex_tuple_2]
				face = list(map(vertex_table.__getitem__, vertex_tuples))
				faces.append(face)
				normals.append(normal_array)
			elif correlation < 0:
				vertex_tuples = [vertex_tuple_0, vertex_tuple_2, vertex_tuple_1]
				face = list(map(vertex_table.__getitem__, vertex_tuples))
				faces.append(face)
				normals.append(-normal_array)
			else:
				vertex_tuples = [vertex_tuple_0, vertex_tuple_2, vertex_tuple_1]
				face = list(map(vertex_table.__getitem__, vertex_tuples))
				faces.append(face)
				normals.append(normal_array)
				vertex_tuples = [vertex_tuple_0, vertex_tuple_1, vertex_tuple_2]
				face = list(map(vertex_table.__getitem__, vertex_tuples))
				faces.append(face)
				normals.append(-normal_array)
		# SET ATTRIBUTES
		self.vertecies    = vertecies
		self.faces        = faces
		self.face_normals = normals


	@blue.restrict
	def save(self, file: str) -> None:
		"""
		This method saves the Mesh data to a file.
		
		Parameters
		----------
		file : str
			The name to which the file is saved.
		
		Raises
		------
		NotImplemented
			If the file format is not supported an error is raised.
		"""
		if file.lower().endswith('.obj'):
			self._save_OBJ(file)
		elif file.lower().endswith('.stl'):
			self._save_STL(file) 
		elif file.lower().endswith('.msh'):
			raise NotImplemented
		else:
			raise NotImplemented


	@blue.restrict
	def _save_OBJ(self, filename: str|None) -> None:
		"""
		This method saves the Mesh data to an ascii obj file.
		
		Parameters
		----------
		filename : str | None
			The name to which the file is saved.
		"""
		with open(self.filename, 'r') as file:
			origin_file = file.read()
		vertex_flag, face_flag = False, False
		with open(filename, 'w') as file:
			file.write('# Exported with microcosm AI blueprints\n')
			file.write('\n# VERTECIES\n')
			for vertex in self.vertecies:
				file.write('v ' + ' '.join(map(lambda x: format(round(float(x), 6), 'f'), vertex)) + '\n')
			if self.vertex_normals:
				file.write('\n# NORMALS\n')
				for normal in self.vertex_normals:
					file.write('vn ' + ' '.join(map(lambda x: format(round(float(x), 6), 'f'), normal)) + '\n')
			if self.texcoords:
				file.write('\n# TEXTURE COORDINATES\n')
				for texcoord in self.texcoords:
					file.write('vt ' + ' '.join(map(lambda x: format(round(float(x), 6), 'f'), texcoord)) + '\n')
			if self.faces:
				file.write('\n# FACES\n')
				for i, face in enumerate(self.faces):
					if i in self.texcoords_idx and i in self.normals_idx:
						file.write('f ' + ' '.join(map(lambda x: f"{x[0] + 1}/{x[1] + 1}/{x[2] + 1}", zip(face, self.texcoords_idx[i], self.normals_idx[i]))) + '\n')
					elif i in self.texcoords_idx:
						file.write('f ' + ' '.join(map(lambda x: f"{x[0] + 1}/{x[1] + 1}", zip(face, self.texcoords_idx[i]))) + '\n')
					elif i in self.normals_idx:
						file.write('f ' + ' '.join(map(lambda x: f"{x[0] + 1}//{x[1] + 1}", zip(face, self.normals_idx[i]))) + '\n')
					else:
						file.write('f ' + ' '.join(map(lambda x: f"{x + 1}", face)) + '\n')


	@blue.restrict
	def _save_STL(self, filename: str) -> None:
		"""
		This method saves the Mesh data to an binary stl file.
		
		Parameters
		----------
		filename : str
			The name to which the file is saved.
		"""
		# GET VERTECIES AND CONSTRUCT CONVERSION FUNCTION
		vertecies      = self.vertecies
		float_to_bytes = lambda x: struct.pack('f', x)
		vector_bytes   = lambda x: float_to_bytes(x[0]) + float_to_bytes(x[1]) + float_to_bytes(x[2])			
		# CREATE HEADER
		SOURCE = bytes('Saved from Blueprints, UNITS= m', encoding='ascii')
		HEADER = bytearray(84)
		HEADER[:len(SOURCE)] = SOURCE
		HEADER[80:] = struct.pack('I', len(self.faces))
		with open(filename,  'wb') as file:
			file.write(HEADER)
			for normal, indecies in zip(self.face_normals, self.faces):
				vertex_a, vertex_b, vertex_c = map(vertecies.__getitem__, indecies)
				file.write(vector_bytes(normal))
				file.write(vector_bytes(vertex_a))
				file.write(vector_bytes(vertex_b))
				file.write(vector_bytes(vertex_c))
				file.write(bytes.fromhex('00') * 2)

	# MUJOCO PROPERTIES

	@property
	def file(self) -> str | None:
		"""
		Filename for Mujoco XML

		Returns
		-------
		str
		"""
		return self.filename


	@property
	def vertex(self):
		"""
		Returns
		-------
		np.ndarray
			This attribute is used to write construct the mujoco xml. It returns a flattened list of 
			coordinates that is written raw to xml. For a structured list of vertecies see :attr:`vertecies`.
		"""
		return self._vertecies.flatten()


	@property
	def face(self) -> np.ndarray|None:
		"""
		Returns
		-------
		np.ndarray | None
			This attribute is used to write construct the mujoco xml. It returns a flattened list of 
			coordinates that is written raw to xml. For a structured list of faces see :attr:`faces`.
		"""
		return self._faces


	@face.setter
	@blue.restrict
	def face(self, face: list) -> None:
		"""
		This attribute is used to write construct the mujoco xml. It returns a flattened list of 
		coordinates that is written raw to xml. For a structured list of faces see :attr:`faces`.
		
		Parameters
		----------
		face : list
			A flattened list of all face indecies.
		"""
		self.faces = face


	@property
	def texcoord(self) -> np.ndarray|None:
		"""
		Returns
		-------
		np.ndarray | None
			This attribute is used to write to xml. It returns a flattened list of texture coordinates that 
			is written raw to xml. For a structured list of texture coordinates see :attr:`texcoords`.
		"""
		return self._texcoords or None


	@texcoord.setter
	@blue.restrict
	def texcoord(self, texcoord: list) -> None:
		"""
		This attribute is used to write to xml. It returns a flattened list of texture coordinates that 
		is written raw to xml. For a structured list of texture coordinates see :attr:`texcoords`.
		
		Parameters
		----------
		texcoord : list
			A flattened list of all texture coordinates.
		"""
		self.texcoords = texcoord


	@property
	def normal(self) -> np.ndarray|None:
		"""
		Returns
		-------
		np.ndarray | None
			This attribute is used to write to xml. It returns a flattened list of face normals that 
			is written raw to xml. For a structured list of face normals see :attr:`normals`.
		"""
		if self._face_normals is None:
			return None
		else:
			return np.concatenate(self._face_normals)


	@normal.setter
	@blue.restrict
	def normal(self, normal: list) -> None:
		"""
		This attribute is used to write to xml. It returns a flattened list of face normals that is 
		written raw to xml. For a structured list of face normals see :attr:`normals`.
		
		Parameters
		----------
		normal : list
			A flattened list of all normals coordinates.
		"""
		self.face_normals = normal


	@property
	def size(self) -> np.ndarray:
		"""
		Returns
		-------
		np.ndarray
			The size of the Mesh is computed as the difference of the minimal and maximal component 
			for each axis on each vertex.
		"""
		return self.vertecies_max - self.vertecies_min
		

	@size.setter
	@blue.restrict
	def size(self, size: np.ndarray|list[int|float]) -> None:
		"""
		The size of the Mesh is defined as the difference of the minimal and maximal component for each 
		axis on each vertex.
		
		Parameters
		----------
		size : np.ndarray | list[int | float]
			Setting this attribute results in a rescaling of all vertecies. Setting this attribute 
			should only be done if the raw data must be modified directly, use :attr:`scale <blueprints.assets.MeshAsset.scale>`
			instead.
		"""
		vertecies = np.array(self.vertecies)
		vertecies = vertecies - self.vertecies_center
		scale     = np.array(size) / self.size
		scale[np.isinf(scale)] = 0
		vertecies = vertecies * scale
		self.vertecies = vertecies + self.vertecies_center

	# BLUEPRINTS PROPERTIES

	@property
	def vertecies(self) -> np.ndarray:
		"""
		Returns
		-------
		np.ndarray
			The list of vertecies. Each vertex is a np.ndarray with 3 components for each of the 
			spatial dimensions.
		"""
		return self._vertecies


	@vertecies.setter
	@blue.restrict
	def vertecies(self, vertecies: np.ndarray|list[np.ndarray|list[float|int]]) -> None:
		"""
		Parameters
		----------
		vertecies : np.ndarray | list[np.ndarray | list[float | int]]
			The list of vertecies. Each vertex is a np.ndarray with 3 components for each of the 
			spatial dimensions.
		"""
		vertecies = np.array(vertecies)
		self._vertecies = vertecies
		self._flag_dependencies('vertecies')
		self._built = False



	@property
	def faces(self) -> list[np.ndarray]|None:
		"""
		Returns
		-------
		list[np.ndarray] | None
			The list of faces. Each face is a np.ndarray with n indecies for each of the vertecies 
			of the face.
		"""
		return self._faces
		####
		if self._faces is None:
			return None
		else:
			return list(np.rollaxis(self._faces.reshape((-1, 3)), axis=0))


	@faces.setter
	@blue.restrict
	def faces(self, faces: np.ndarray|list[np.ndarray|list[float|int]]|None) -> None:
		"""
		Parameters
		----------
		faces : np.ndarray | list[np.ndarray | list[float | int]] | None
			The list of faces. Each face is a np.ndarray with n indecies for each of the vertecies 
			of the face.
		"""
		if isinstance(faces, np.ndarray):
			faces = list(np.rollaxis(faces, axis=0))
		self._faces = faces
		self._built = False
		return
		###
		if faces is not None:
			faces = np.array(faces, dtype=np.int32).flatten()
		self._faces = faces
		self._built = False


	@property
	def texcoords(self) -> list[np.ndarray]|None:
		"""
		Returns
		-------
		list[np.ndarray] | None
			The list of texture coordinates. Each texture coordinate is a np.ndarray with 
			coordinates in the range [0, 1] specifying the point on the texture which is mapped to
			the referencing face vertex.
		"""
		if self._texcoords is None:
			return None
		else:
			return list(np.rollaxis(self._texcoords.reshape((-1, 2)), axis=0))


	@texcoords.setter
	@blue.restrict
	def texcoords(self, texcoords: np.ndarray|list[np.ndarray|list[float|int]]|None) -> None:
		"""
		Parameters
		----------
		texcoords : np.ndarray | list[np.ndarray | list[float | int]] | None
			The list of texture coordinates. Each texture coordinate is a np.ndarray with 
			coordinates in the range [0, 1] specifying the point on the texture which is mapped to
			the referencing face vertex.
		"""
		if texcoords is not None:
			texcoords = np.array(texcoords, dtype=np.float32).flatten()
		self._texcoords = texcoords
		self._built = False


	@property
	def texcoords_idx(self) -> dict|None:
		"""
		Returns
		-------
		dict | None
			The list of texture coordinate indecies for faces.
		"""
		return self._texcoords_idx


	@texcoords_idx.setter
	@blue.restrict
	def texcoords_idx(self, texcoords_idx: dict|None) -> None:
		"""
		Parameters
		----------
		texcoords_idx : dict | None
			The list of texture coordinate indecies for faces.
		"""
		self._texcoords_idx = texcoords_idx
		self._built = False


	@property
	def face_normals(self) -> list[np.ndarray]|None:
		"""
		Returns
		-------
		list[np.ndarray] | None
			Face normals are the normal vectors of faces. They specify the direction in which the 
			face points.
		"""
		if self._face_normals is None:
			if self._faces is not None:
				faces = self.vertecies[self.faces,:]
				edges = faces[:,:,1:] - faces[:,:,:1]
				cross = np.cross(edges[...,0], edges[...,1], axisa=1, axisb=1)
				norms = cross / np.linalg.norm(cross, axis=1)[:,None]
				self.face_normals = norms
				return self.face_normals
			else:
				return None
		else:
			return self._face_normals


	@face_normals.setter
	@blue.restrict
	def face_normals(self, face_normals: np.ndarray|list[np.ndarray|list[float|int]]|None) -> None:
		"""
		Parameters
		----------
		face_normals : np.ndarray | list[np.ndarray | list[float | int]] | None
			Face normals are the normal vectors of faces. They specify the direction in which the 
			face points.
		"""
		if isinstance(face_normals, np.ndarray):
			face_normals = list(np.rollaxis(face_normals, axis=0))
		self._face_normals = face_normals
		self._built = False


	@property
	def vertex_normals(self) -> list[np.ndarray]|None:
		"""
		Returns
		-------
		list[np.ndarray] | None
			Vertex normals are the normal vectors of vertecies. They specify the direction in which 
			the vertecies of a face point. Specifying vertex normals instead of face normals enables 
			mujoco torenderer soft edges. 
		"""

		if self._vertex_normals is None:
			return None
		else:
			return self._vertex_normals


	@vertex_normals.setter
	@blue.restrict
	def vertex_normals(self, vertex_normals: np.ndarray|list[np.ndarray|list[float|int]]|None) -> None:
		"""
		Parameters
		----------
		vertex_normals : np.ndarray | list[np.ndarray | list[float | int]] | None
			Vertex normals are the normal vectors of vertecies. They specify the direction in which 
			the vertecies of a face point. Specifying vertex normals instead of face normals enables 
			mujoco torenderer soft edges. 
		"""
		if isinstance(vertex_normals, np.ndarray):
			vertex_normals = list(np.rollaxis(vertex_normals, axis=0))
		self._vertex_normals = vertex_normals
		self._built = False


	@property
	def normals_idx(self) -> dict|None:
		"""
		The list of normals indecies for faces.
		
		Returns
		-------
		dict | None
		"""
		return self._normals_idx


	@normals_idx.setter
	@blue.restrict
	def normals_idx(self, normals_idx: dict|None) -> None:
		"""
		Parameters
		----------
		normals_idx : dict | None
			The list of normals indecies for faces.
		"""
		self._normals_idx = normals_idx
		self._built = False


	@property
	def vertecies_min(self):
		"""
		The minimum for each axis component of all vertecies.
		
		Returns
		-------
		np.ndarray
		"""
		if self._validate('vertecies_min'):
			self._vertecies_min = np.min(self.vertecies, axis=0)
			self._DEPENDENCY_FLAGS['vertecies_min'] = True
		return self._vertecies_min


	@property
	def vertecies_max(self):
		"""
		The maximum for each axis component of all vertecies.
		
		Returns
		-------
		np.ndarray
		"""
		if self._validate('vertecies_max'):
			self._vertecies_max = np.max(self.vertecies, axis=0)
			self._DEPENDENCY_FLAGS['vertecies_max'] = True
		return self._vertecies_max


	@property
	def vertecies_center(self):
		"""
		The middle point for each axis component of all vertecies.
		
		Returns
		-------
		np.ndarray
		"""
		return (self.vertecies_min + self.vertecies_max)/2



class HFieldCache(blue.HFieldCacheType, BaseCache):
	def __init__(self, 
		     terrain:  np.ndarray|list[np.ndarray|list[float|int]]|None = None,  
		     filename: str|None = None, 
		     **kwargs):
		self._built = False
		self._world = None
		super().__init__(**kwargs)
		self.filename = filename
		if filename is not None:
			self.load(filename)
		elif terrain is not None:
			self.terrain = terrain
		if 'name' not in kwargs:
			self._name = None


	def __getitem__(self, 
			key: tuple[slice|int]) -> np.ndarray|np.float32:
		"""
		Returns
		-------
		np.ndarray | np.float32
			The values from the selected index/slice of the height field.
		"""
		return self.terrain[key]


	def __setitem__(self, 
			key:   tuple[slice|int], 
			value: int|float|list[int|float]|np.ndarray) -> None:
		"""
		Parameters
		----------
		key : tuple[slice | int]
			The indecies/slices of acces
		value : int|float|list[int|float]|np.ndarray
			The value to be assigned in the selected parts of the field.
		"""
		self.terrain[key] = np.array(value, dtype=np.float32)


	@blue.restrict
	def copy(self, **kwargs) -> blue.CacheType:
		"""
		This method constructs a copy of the Cache with possible alterations to its 
		attributes as specified in kwargs.
		
		Parameters
		----------
		**kwargs
			Keyword arguments are passed to the :meth:`__init__` to replace the attributes 
			of the current Thing from which copy is called.
		
		Returns
		-------
		blue.CacheType
			A new instance of the Cache
		"""
		mesh = super().copy(**kwargs)
		if 'name' not in kwargs:
			mesh._name = self._name
		return mesh


	@blue.restrict
	def _build(self, dirname: str, **kwargs) -> None:
		"""
		This method is called by the Assets parent to construct the xml representations of 
		the kinematic tree.
		
		Parameters
		----------
		dirname : str
			The directory name in which the hfield file is saved.
		**kwargs
			Dummy argument
		"""
		if not self._built:
			filename = self.filename or f'hfield.hf'
			pathname, basename = os.path.split(filename)
			if not pathname.endswith(dirname):
				path = f'{dirname}/{self.ID}_{basename}'
			else:
				path = filename
			self.save(path)
			self._path = path
			self._built = True


	@blue.restrict
	@classmethod
	def _from_xml_element(cls, xml_element) -> blue.CacheType:
		"""
		This method reconstructs an HFieldCache from an xml element.
		
		Parameters
		----------
		xml_element : xml.Element
			The xml element from which a HFieldCache is reconstructed.
		
		Returns
		-------
		blue.CacheType
			The reconstructed HFieldCache.
		"""
		init_args, post_args, rest_args = cls._xml_element_args(xml_element)
		if 'terrain' in post_args:
			init_args['terrain'] = post_args['terrain']
			del post_args['terrain']
		obj = object.__new__(cls)
		obj.__init__(**init_args)
		for key, val in post_args.items():
			setattr(obj, key, val)
		return obj


	@blue.restrict
	def load(self, filename: str) -> None:
		"""
		This methods loads the mesh data from a file.
		
		Parameters
		----------
		filename : str
			Possible file types are ``'.stl'`` binary or ascii and ``'.obj'`` ascii.
		"""
		if filename.lower().endswith('.hf'):
			with open(filename, 'rb') as file:
				data = file.read()
			self._load_HF(data)
		elif filename.lower().endswith('.png'):
			self._load_PNG(filename)
		else:
			raise Exception('HFields are only implemented for PNG and HF.')


	@blue.restrict
	def _load_HF(self, data: bytes) -> None:
		"""
		Parses the data from an binary stl file to the Cache.
		
		Parameters
		----------
		data : bytes
			The data of the file to be parsed
		"""
		# CONVERSION FUNCTIONS
		bytes_to_float = lambda x: struct.unpack('f', x)[0]
		vector_tuple   = lambda x: tuple(map(bytes_to_float, (x[:4], x[4:8], x[8:])))
		# HEADER DATA
		nrows          = struct.unpack('I', data[0:4])[0]
		ncols          = struct.unpack('I', data[4:8])[0]
		# TRTIANGLE DATA
		height_data    = data[8:]
		rows           = []
		for i in range(nrows * ncols):
			height = height_data[i*4:(i+1)*4]
			heights.append(bytes_to_float(height))
		# SET ATTRIBUTES
		self.terrain    = np.array(heights).reshape((nrows, ncols))
		

	@blue.restrict
	def _load_PNG(self, filename: str) -> None:
		image        = imread(filename)
		height       = np.mean(image[:,:,:3])
		self.terrain = height
	

	@blue.restrict
	def save(self, filename: str) -> None:
		"""
		This method saves the Mesh data to a file.
		
		Parameters
		----------
		file : str
			The name to which the file is saved.
		
		Raises
		------
		NotImplemented
			If the file format is not supported an error is raised.
		"""
		if filename.lower().endswith('.hf'):
			self._save_HF(filename)
		else:
			raise NotImplemented


	@blue.restrict
	def _save_HF(self, filename: str) -> None:
		"""
		This method saves the Mesh data to an binary stl file.
		
		Parameters
		----------
		filename : str
			The name to which the file is saved.
		"""
		# GET VERTECIES AND CONSTRUCT CONVERSION FUNCTION
		int_to_bytes   = lambda x: struct.pack('I', x)
		float_to_bytes = lambda x: struct.pack('f', x)
		with open(filename, 'wb') as file:
			file.write(int_to_bytes(self.nrow))
			file.write(int_to_bytes(self.ncol))
			for height in self.terrain.reshape(-1):
				file.write(float_to_bytes(height))
	
	# DERIVED PROPERTIES

	@property
	def elevation(self):
		"""
		Derived dummy property
		"""
		return self.terrain.flatten()

	@property
	def nrow(self) -> int:
		"""
		Number of rows in the terrain.

		Returns
		-------
		int
		"""
		return int(self.terrain.shape[0])

	@property
	def ncol(self) -> int:
		"""
		Number of collumns in the terrain.

		Returns
		-------
		int
		"""
		return int(self.terrain.shape[1])

	# BLUEPRINTS PROPERTIES

	@property
	def filename(self) -> str:
		"""
		The filename of the hfield data.

		Returns
		-------
		str
		"""
		return self._filename


	@filename.setter
	@blue.restrict
	def filename(self, filename: str|None) -> None:
		self._filename = filename
	

	@property
	def terrain(self) -> np.ndarray:
		"""
		Returns
		-------
		np.ndarray
			The list of vertecies. Each vertex is a np.ndarray with 3 components for each of the 
			spatial dimensions.
		"""
		return self._terrain


	@terrain.setter
	@blue.restrict
	def terrain(self, terrain: np.ndarray|list[np.ndarray|list[float|int]]) -> None:
		"""
		Parameters
		----------
		vertecies : np.ndarray | list[np.ndarray | list[float | int]]
			The list of vertecies. Each vertex is a np.ndarray with 3 components for each of the 
			spatial dimensions.
		"""
		self._terrain = np.array(terrain, dtype=np.float32)
		if self._world is not None:
			mj_hfield = self._world._mj_data.model.hfield(self._index)
			mj_hfield.data = self._terrain
			if self._world._viewer is not None:
				self._world._viewer.update_hfield(self._index)
			#self._world._mj_model.update_hfield(self._index)


	@property
	def file(self) -> str:
		"""
		Dummy variable for the default names of hfield data files.

		Returns
		-------
		str
		"""
		if self._file is None:
			return f'hfield_{self.name}.hf'
		else:
			return self._file
	
	
		





CACHE_THINGS = {'mesh':   MeshCache, 
		'hfield': HFieldCache}