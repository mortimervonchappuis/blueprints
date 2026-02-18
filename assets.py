"""
.. note::
	It is recommended to not use assets directly. All property access is 
	handeled through a front facing ThingType. Assets are :mod:`UniqueThings <blueprints.thing.unique>`.
"""



import os
import struct
import xml.etree.ElementTree as xml
import numpy as np
import blueprints as blue
from blueprints.thing.colored import Color
from collections import defaultdict
from itertools import count
from imageio import imread, imwrite



ASSET_DIR = 'assets'



class BaseAsset(blue.AssetType, blue.UniqueThing):
	"""
	Assets are used to link external resources to the model. Currently only :class:`MeshAssets <MeshAsset>`, 
	 :class:`HFieldAssets <HFieldAsset>`, :class:`TextureAssets <TextureAsset>`, 
	 :class:`MaterialAssets <MaterialAsset>` are supported, but ``SkinAssets`` and 
	``BoneAssets`` will be developed in a future version version.
	"""
	@property
	def name(self) -> str:
		"""
		The user specified name might potentially be altered to avoid a naming conflict by appending an 
		enumeration scheme.
		
		Returns
		-------
		str
		"""
		return self._name or f'asset_{self.ID}'


	@name.setter
	@blue.restrict
	def name(self, name: str) -> None:
		"""
		The user specified name might potentially be altered to avoid a naming conflict by appending an 
		enumeration scheme.
		
		Parameters
		----------
		name : str
			The user specified name.
		"""
		self._name = name



class MaterialAsset(blue.MaterialAssetType, BaseAsset, blue.thing.ColoredThing):
	"""
	See :mod:`Material <blueprints.material>`
	
	Most attribute descriptions are partially taken from `Mujoco <https://mujoco.readthedocs.io/en/latest/XMLreference.html#asset-material>`__.
	"""
	@blue.restrict
	def __init__(self, 
		     texture:     blue.TextureType|None = None, 
		     texrepeat:   list[int|float]|np.ndarray = [1, 1], 
		     texuniform:  bool      = False, 
		     emission:    int|float = 0.0, 
		     specular:    int|float = 0.5, 
		     shininess:   int|float = 0.5, 
		     reflectance: int|float = 0.0, 
		     metallic:    int|float =-1.0, 
		     roughness:   int|float =-1.0, 
		     color:       str|object|None = None, 
		     name:        str|None        = None):
		self._built      = False
		# MUJOCO ATTRIBUTES
		self.texrepeat   = texrepeat
		self.texuniform  = texuniform
		self.emission    = emission
		self.specular    = specular
		self.shininess   = shininess
		self.reflectance = reflectance
		self.metallic    = metallic
		self.roughness   = roughness
		# TEXTURE
		self.texture     = texture
		super().__init__(name=name)
		if color is not None:
			self.color = color
		if name is None:
			self._name = None


	@blue.restrict
	def _build(self, 
		   parent, 
		   world, 
		   indicies, 
		   **kwargs):
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
		if not self._built:
			self._built = True
			if self.texture is not None:
				self._xml_root = xml.SubElement(world._xml_asset, 
							        'material', 
							        texture=self.texture.asset.name, 
								**self._mujoco_specs(kwargs))
			else:
				self._xml_root = xml.SubElement(world._xml_asset, 
							        'material', 
								**self._mujoco_specs(kwargs))
			#self._index       = indicies['mesh']
			#indicies['mesh'] += 1
			if self.texture is not None:
				self.texture._build(parent=parent, 
						    world=world, 
						    indicies=indicies, 
						    **kwargs)
		return self._xml_root



	@property
	def texrepeat(self) -> np.ndarray:
		"""
		The number of repetitions of the Texture in two directions

		Returns
		-------
		np.ndarray
		"""
		return self._texrepeat.copy()


	@texrepeat.setter
	@blue.restrict
	def texrepeat(self, texrepeat: list[int|float]|np.ndarray) -> None:
		if isinstance(texrepeat, list):
			texrepeat = np.array(texrepeat, dtype=np.float32)
		self._texrepeat = texrepeat


	@property
	def texuniform(self) -> bool:
		"""
		For :class:`BoxTextures <blueprints.texture.Box>`, this attribute controls how 
		cube mapping is applied. The default value ``False`` means apply cube mapping 
		directly, using the actual size of the object. The value “true” maps the texture 
		to a unit object before scaling it to its actual size (geometric primitives are 
		created by the renderer as unit objects and then scaled). In some cases this 
		leads to more uniform texture appearance, but in general, which settings produces 
		better results depends on the texture and the object. For 2d textures, this attribute 
		interacts with texrepeat above. Let texrepeat be N. The default value “false” means 
		that the 2d texture is repeated N times over the (z-facing side of the) object. 
		The value “true” means that the 2D :class:`PlaneTexture <blueprints.texture.Plane>` 
		is repeated N times over one spatial unit, regardless of object size.

		Returns
		-------
		bool
		"""
		return self._texuniform


	@texuniform.setter
	@blue.restrict
	def texuniform(self, texuniform: bool) -> None:
		self._texuniform = texuniform


	@property
	def emission(self) -> float:
		"""
		If set positive, the Material emitts lights.
		
		Returns
		-------
		float
		"""
		return self._emission


	@emission.setter
	@blue.restrict
	def emission(self, emission: int|float) -> None:
		self._emission = float(emission)


	@property
	def specular(self) -> float:
		"""
		Specularity in OpenGL. This value should be in the range [0 1].

		Returns
		-------
		float
		"""
		return self._specular


	@specular.setter
	@blue.restrict
	def specular(self, specular: int|float) -> None:
		self._specular = float(specular)


	@property
	def shininess(self) -> float:
		"""
		Shininess in OpenGL.

		Returns
		-------
		float
		"""
		return self._shininess


	@shininess.setter
	@blue.restrict
	def shininess(self, shininess: int|float) -> None:
		self._shininess = float(shininess)


	@property
	def reflectance(self) -> float:
		"""
		This attribute should be in the range [0 1]. If the value 
		is greater than 0, and the material is applied to a plane 
		or a box geom, the renderer will simulate reflectance. The 
		larger the value, the stronger the reflectance. For boxes, 
		only the face in the direction of the local +Z axis is 
		reflective. Simulating reflectance properly requires 
		ray-tracing which cannot (yet) be done in real-time. We 
		are using the stencil buffer and suitable projections instead. 
		Only the first reflective geom in the model is rendered as 
		such. This adds one extra rendering pass through all geoms, 
		in addition to the extra rendering pass added by each 
		shadow-casting light.

		Returns
		-------
		float
		"""
		return self._reflectance


	@reflectance.setter
	@blue.restrict
	def reflectance(self, reflectance: int|float) -> None:
		self._reflectance = float(reflectance)


	@property
	def metallic(self) -> float:
		"""
		This attribute corresponds to uniform metallicity 
		coefficient applied to the entire material. This 
		attribute has no effect in MuJoCo’s native renderer, 
		but it can be useful when rendering scenes with a 
		physically-based renderer. In this case, if a 
		non-negative value is specified, this metallic value 
		should be multiplied by the metallic texture sampled 
		value to obtain the final metallicity of the material.

		Returns
		-------
		float
		"""
		return self._metallic


	@metallic.setter
	@blue.restrict
	def metallic(self, metallic: int|float) -> None:
		self._metallic = float(metallic)


	@property
	def roughness(self) -> float:
		"""
		This attribute corresponds to uniform roughness 
		coefficient applied to the entire material. This 
		attribute has no effect in MuJoCo’s native renderer, 
		but it can be useful when rendering scenes with a 
		physically-based renderer. In this case, if a non-negative 
		value is specified, this roughness value should be 
		multiplied by the roughness texture sampled value to 
		obtain the final roughness of the material.

		Returns
		-------
		float
		"""
		return self._roughness


	@roughness.setter
	@blue.restrict
	def roughness(self, roughness: int|float) -> None:
		self._roughness = float(roughness)



class TextureAsset(blue.TextureAssetType, BaseAsset):
	"""
	See :mod:`Texture <blueprints.texture>`
	
	Most attribute descriptions are partially taken from `Mujoco <https://mujoco.readthedocs.io/en/latest/XMLreference.html#asset-texture>`__.
	"""
	@blue.restrict
	def __init__(self, 
		     #color_space: str       = 'auto', 
		     content:     str|None  =  None, 
		     file:        str|None  =  None, 
		     grid_size:   list[int] = [1, 1], 
		     grid_layout: str|None  =  None,
		     fileright:   str|None  =  None, 
		     fileleft:    str|None  =  None, 
		     fileup:      str|None  =  None, 
		     filedown:    str|None  =  None, 
		     filefront:   str|None  =  None, 
		     fileback:    str|None  =  None, 
		     builtin:     str|None  =  None, 
		     color_1:     object    = [0.8, 0.8, 0.8], 
		     color_2:     object    = [0.5, 0.5, 0.5], 
		     mark:        str|None  =  None, 
		     color_mark:  object    = [0.0, 0.0, 0.0],
		     random:      float     =  0.01, 
		     width:       int       =  0, 
		     height:      int       =  0, 
		     h_flip:      bool      =  False, 
		     v_flip:      bool      =  False, 
		     n_channel:   int       =  3, 
		     name:        str|None  = None):
		self._built       = False
		self._image       = None
		self._image_up    = None
		self._image_down  = None
		self._image_left  = None
		self._image_right = None
		self._image_front = None
		self._image_back  = None
		#self.color_space = color_space
		self.content      = content
		self.file         = file
		self.grid_size    = grid_size
		self.grid_layout  = grid_layout
		self.fileright    = fileright
		self.fileleft     = fileleft
		self.fileup       = fileup
		self.filedown     = filedown
		self.filefront    = filefront
		self.fileback     = fileback
		self.builtin      = builtin
		self.color_1      = color_1
		self.color_2      = color_2
		self.mark         = mark
		self.color_mark   = color_mark
		self.random       = random
		self.width        = width
		self.height       = height
		self.h_flip       = h_flip
		self.v_flip       = v_flip
		self.n_channel    = n_channel
		super().__init__(name=name)
		self._load_images()
		if name is None:
			self._name = None


	@blue.restrict
	def _build(self, 
		   parent, 
		   world, 
		   indicies, 
		   **kwargs):
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
		if not self._built:
			# MAKE DIRECTORIES
			assets_path = f'{world._path}/{ASSET_DIR}'
			if not os.path.exists(assets_path):
				os.mkdir(assets_path)
			dirname = f'{world._path}/{ASSET_DIR}/textures'
			if not os.path.exists(dirname):
				os.mkdir(dirname)
			# SAVE FILES
			filenames = self._save_images(dirname)
			# UPDATE FILENAMES
			kwargs = kwargs.copy()
			kwargs.update(filenames)
			self._built = True
			self._xml_root = xml.SubElement(world._xml_asset, 
							'texture', 
						        **self._mujoco_specs(kwargs))
		return self._xml_root


	def _load_images(self):
		if self.file       is not None:
			self._image       = imread(self.file)
		if self.fileup    is not None:
			self._imageup    = imread(self.fileup)
		if self.filedown  is not None:
			self._imagedown  = imread(self.filedown)
		if self.fileleft  is not None:
			self._imageleft  = imread(self.fileleft)
		if self.fileright is not None:
			self._imageright = imread(self.fileright)
		if self.filefront is not None:
			self._imagefront = imread(self.filefront)
		if self.fileback  is not None:
			self._imageback  = imread(self.fileback)


	def _save_images(self, dirname) -> dict:
		files = dict()
		if self.file       is not None:
			file       = self._resolve_filename(dirname, self.file)
			imwrite(file, self._image)
			files['file']       = file
		if self.fileup    is not None:
			fileup    = self._resolve_filename(dirname, self.fileup)
			imwrite(fileup, self._imageup)
			files['fileup']    = fileup
		if self.filedown  is not None:
			filedown  = self._resolve_filename(dirname, self.filedown)
			imwrite(filedown, self._imagedown)
			files['filedown']  = filedown
		if self.fileleft  is not None:
			fileleft  = self._resolve_filename(dirname, self.fileleft)
			imwrite(fileleft, self._imageleft)
			files['fileleft']  = fileleft
		if self.fileright is not None:
			fileright = self._resolve_filename(dirname, self.fileright)
			imwrite(fileright, self._imageright)
			files['fileright'] = fileright
		if self.filefront is not None:
			filefront = self._resolve_filename(dirname, self.filefront)
			imwrite(filefront, self._imagefront)
			files['filefront'] = filefront
		if self.fileback  is not None:
			fileback  = self._resolve_filename(dirname, self.fileback)
			imwrite(fileback, self._imageback)
			files['fileback']  = fileback
		return files


	def _resolve_filename(self, dirname: str, filename: str, prefix=None):
		pathname, basename = os.path.split(filename)
		if not pathname.endswith(dirname):
			if prefix is None:
				return f'{dirname}/{self.ID}_{basename}'
			else:
				return f'{dirname}/{self.ID}_{prefix}_{basename}'
		else:
			return filename

	# DERIVED ATTRIBUTES

	@property
	def type(self) -> str:
		"""
		Derived type attribute for Mujoco,

		Returns
		-------
		str
		"""
		return self._TYPE


	#@property
	#def colorspace(self) -> str:
	#	return self.color_space


	@property
	def content_type(self) -> str|None:
		"""
		Content type (MIME type) taking values ``None``, ``'image/png'``, 
		``'image/ktx'`` and ``'image/vnd.mujoco.texture'``.

		Returns
		-------
		str | None
		"""
		return self.content


	@property
	def gridsize(self) -> list[int]:
		"""
		Grid size attrigute ``[n_rows, n_cols]`` defining the grid. 
		For non-grid Textures this value is ignored.

		Returns
		-------
		list[int]
		"""
		return np.array(self.grid_size, dtype=np.int8)


	@property
	def gridlayout(self) -> str|None:
		"""
		The layout of the grid. See example above.

		Returns
		-------
		str | None
		"""
		return self.grid_layout


	@property
	def rgb1(self) -> np.ndarray:
		"""
		The first color of a procedural Texture.

		Returns
		-------
		blue.ColorType
		"""
		return self.color_1.rgb


	@property
	def rgb2(self) -> np.ndarray:
		"""
		The second color of a procedural Texture.
		
		Returns
		-------
		blue.ColorType
		"""
		return self.color_2.rgb


	@property
	def markrgb(self) -> np.ndarray:
		"""
		The mark color of a procedural Texture.
		
		Returns
		-------
		blue.ColorType
		"""
		return self.color_mark.rgb


	@property
	def hflip(self) -> bool:
		"""
		If true, images loaded from file are flipped in the horizontal 
		direction. Does not affect procedural textures.

		Returns
		-------
		bool
		"""
		return self.h_flip


	@property
	def vflip(self) -> bool:
		"""
		If true, images loaded from file are flipped in the vertical 
		direction. Does not affect procedural textures.

		Returns
		-------
		bool
		"""
		return self.v_flip


	@property
	def nchannel(self) -> int:
		"""
		The number of channels in the texture image file. This allows 
		loading 4-channel textures (RGBA) or single-channel textures 
		(e.g., for Physics-Based Rendering properties such as roughness 
		or metallic).

		Returns
		-------
		int
		"""
		return self.n_channel

	# BLUEPRINTS ATTRIBUTES

	#@property
	#def color_space(self) -> str:
	#	return self._color_space


	#@color_space.setter
	#@blue.restrict
	#def color_space(self, color_space: str) -> None:
	#	color_space = color_space.strip()
	#	if color_space not in ('auto', 'linear', 'sRGB'):
	#		raise ValueError(f'Attribute color_space must be set to "auto", "linear" or "sRGB". Got "{color_space}" instead.')
	#	self._color_space = color_space


	@property
	def content(self) -> str|None:
		"""
		Content type (MIME type) taking values ``None``, ``'image/png'``, 
		``'image/ktx'`` and ``'image/vnd.mujoco.texture'``.

		Returns
		-------
		str | None
		"""
		return self._content


	@content.setter
	@blue.restrict
	def content(self, content: str|None) -> None:
		if isinstance(content, str):
			content = content.strip()
		if content is not None and content not in ('image/png', 'image/ktx', 'image/vnd.mujoco.texture'):
			raise ValueError(f'Attribute content must be set to MIME type "image/png", "image/ktx" or "image/vnd.mujoco.texture". Got "{content}" instead.')
		self._content = content


	@property
	def file(self) -> str|None:
		"""
		Name of the texture file.

		Returns
		-------
		str | None
		"""
		return self._file


	@file.setter
	@blue.restrict
	def file(self, file: str|None) -> None:
		if isinstance(file, str):
			file = file.strip()
		self._file = file


	@property
	def grid_size(self) -> list[int]:
		"""
		Grid size attrigute ``[n_rows, n_cols]`` defining the grid. 
		For non-grid Textures this value is ignored.

		Returns
		-------
		list[int]
		"""
		return [self._n_rows, self._n_cols]


	@grid_size.setter
	@blue.restrict
	def grid_size(self, grid_size: list[int]) -> None:
		if len(grid_size) != 2:
			raise ValueError(f'grid_size must be a list of length 2. Got {len(grid_size)} instead.')
		rows, cols = grid_size
		if rows < 0:
			raise ValueError(f'Rows in grid_size[0] must be positive. Got {rows} instead.')
		if cols < 0:
			raise ValueError(f'Columns in grid_size[1] must be positive. Got {cols} instead.')
		if rows * cols > 12:
			raise ValueError(f'Rows × Cols in grid_size must not multiply beyond 12. Got {rows} × {cols} = {rows * cols} instead.')
		self._n_rows = rows
		self._n_cols = cols


	@property
	def grid_layout(self) -> str|None:
		"""
		The layout of the grid. See example above.

		Returns
		-------
		str | None
		"""
		return self._grid_layout


	@grid_layout.setter
	@blue.restrict
	def grid_layout(self, grid_layout: str|None) -> None:
		self._grid_layout = grid_layout


	@property
	def fileright(self) -> str|None:
		"""
		The file name for the right side of a :class:`Box`.

		Returns
		-------
		str | None
		"""
		return self._fileright


	@fileright.setter
	@blue.restrict
	def fileright(self, fileright: str|None) -> None:
		if isinstance(fileright, str):
			fileright = fileright.strip()
		self._fileright = fileright


	@property
	def fileleft(self) -> str|None:
		"""
		The file name for the left side of a :class:`Box`.
		
		Returns
		-------
		str | None
		"""
		return self._fileleft


	@fileleft.setter
	@blue.restrict
	def fileleft(self, fileleft: str|None) -> None:
		if isinstance(fileleft, str):
			fileleft = fileleft.strip()
		self._fileleft = fileleft


	@property
	def fileup(self) -> str|None:
		"""
		The file name for the upper side of a :class:`Box`.
		
		Returns
		-------
		str | None
		"""
		return self._fileup


	@fileup.setter
	@blue.restrict
	def fileup(self, fileup: str|None) -> None:
		if isinstance(fileup, str):
			fileup = fileup.strip()
		self._fileup = fileup


	@property
	def filedown(self) -> str|None:
		"""
		The file name for the down side of a :class:`Box`.
		
		Returns
		-------
		str | None
		"""
		return self._filedown


	@filedown.setter
	@blue.restrict
	def filedown(self, filedown: str|None) -> None:
		if isinstance(filedown, str):
			filedown = filedown.strip()
		self._filedown = filedown


	@property
	def filefront(self) -> str|None:
		"""
		The file name for the front side of a :class:`Box`.
		
		Returns
		-------
		str | None
		"""
		return self._filefront


	@filefront.setter
	@blue.restrict
	def filefront(self, filefront: str|None) -> None:
		if isinstance(filefront, str):
			filefront = filefront.strip()
		self._filefront = filefront


	@property
	def fileback(self) -> str|None:
		"""
		The file name for the back side of a :class:`Box`.
		
		Returns
		-------
		str | None
		"""
		return self._fileback


	@fileback.setter
	@blue.restrict
	def fileback(self, fileback: str|None) -> None:
		if isinstance(fileback, str):
			fileback = fileback.strip()
		self._fileback = fileback


	@property
	def builtin(self) -> str|None:
		"""
		If set, this attribute specifies a builtin procedural Texture. 
		See examples above.

		Returns
		-------
		str | None
		"""
		return self._builtin


	@builtin.setter
	@blue.restrict
	def builtin(self, builtin: str|None) -> None:
		if isinstance(builtin, str):
			builtin = builtin.strip()
		if builtin is not None and builtin not in ('gradient', 'checker', 'flat'):
			raise ValueError(f'Attribute builtin must be set to "gradient", "checker", "flat" or None. Got "{builtin}" instead.')
		self._builtin = builtin


	@property
	def color_1(self) -> Color:
		"""
		The first color of a procedural Texture.

		Returns
		-------
		blue.ColorType
		"""
		return self._color_1


	@color_1.setter
	@blue.restrict
	def color_1(self, color_1: object) -> None:
		self._color_1 = Color(color_1)


	@property
	def color_2(self) -> Color:
		"""
		The second color of a procedural Texture.
		
		Returns
		-------
		blue.ColorType
		"""
		return self._color_2


	@color_2.setter
	@blue.restrict
	def color_2(self, color_2: object) -> None:
		self._color_2 = Color(color_2)


	@property
	def mark(self) -> str|None:
		"""
		If set this attribute specifies additional markings on 
		a procedural Texture. Possible values are ``None``, 
		``'edge'``, ``'cross'`` and ``'random'``. See examples above.
		
		Returns
		-------
		str | None
		"""
		return self._mark


	@mark.setter
	@blue.restrict
	def mark(self, mark: str|None) -> None:
		if isinstance(mark, str):
			mark = mark.strip()
		if mark is not None and mark not in ('edge', 'cross', 'random'):
			raise ValueError(f'Attribute mark must be set to "edge", "cross", "random" or None. Got "{mark}" instead.')
		self._mark = mark


	@property
	def color_mark(self) -> Color:
		"""
		The mark color of a procedural Texture.
		
		Returns
		-------
		blue.ColorType
		"""
		return self._color_mark


	@color_mark.setter
	@blue.restrict
	def color_mark(self, color_mark: object) -> None:
		self._color_mark = Color(color_mark)


	@property
	def random(self) -> float:
		"""
		If the ``mark`` attribute is set to ``'random'``, this 
		attribute specifies the probability of a pixel in the 
		texture being the ``mark_color``.

		Returns
		-------
		float
		"""
		return self._random


	@random.setter
	@blue.restrict
	def random(self, random: float) -> None:
		if not 0 <= random <= 1:
			raise ValueError(f'Attribute random specifies a probability and must lay within the interval [0, 1]. Got {random} instead.')
		self._random = random


	@property
	def width(self) -> int:
		"""
		The width of a procedural texture, i.e., the number of columns 
		in the image. Larger values usually result in higher quality 
		images, although in some cases (e.g. checker patterns) small 
		values are sufficient. For textures loaded from files, this 
		attribute is ignored.

		Returns
		-------
		int
		"""
		return self._width


	@width.setter
	@blue.restrict
	def width(self, width: int) -> None:
		if width < 0:
			raise ValueError(f'Attribute width must be positive. Got {width} instead.')
		self._width = width


	@property
	def height(self) -> int:
		"""
		The height of the procedural texture, i.e., the number of rows 
		in the image. For :class:`Box` and :class:`Skybox` textures, this attribute is 
		ignored and the height is set to 6 times the width. For textures 
		loaded from files, this attribute is ignored.

		Returns
		-------
		int
		"""
		return self._height


	@height.setter
	@blue.restrict
	def height(self, height: int) -> None:
		if height < 0:
			raise ValueError(f'Attribute height must be positive. Got {height} instead.')
		self._height = height


	@property
	def n_channel(self) -> int:
		"""
		The number of channels in the texture image file. This allows 
		loading 4-channel textures (RGBA) or single-channel textures 
		(e.g., for Physics-Based Rendering properties such as roughness 
		or metallic).

		Returns
		-------
		int
		"""
		return self._n_channel


	@n_channel.setter
	@blue.restrict
	def n_channel(self, n_channel: int) -> None:
		if n_channel < 0:
			raise ValueError(f'Attribute n_channel must be positive. Got {n_channel} instead.')
		self._n_channel = n_channel


	@property
	def h_flip(self) -> bool:
		"""
		If true, images loaded from file are flipped in the horizontal 
		direction. Does not affect procedural textures.

		Returns
		-------
		bool
		"""
		return self._h_flip


	@h_flip.setter
	@blue.restrict
	def h_flip(self, h_flip: bool) -> None:
		self._h_flip = h_flip


	@property
	def v_flip(self) -> bool:
		"""
		If true, images loaded from file are flipped in the vertical 
		direction. Does not affect procedural textures.

		Returns
		-------
		bool
		"""
		return self._v_flip


	@v_flip.setter
	@blue.restrict
	def v_flip(self, v_flip: bool) -> None:
		self._v_flip = v_flip



class MeshAsset(blue.MeshAssetType, BaseAsset, blue.thing.MoveableThing):

	"""
	MeshAssets handle access to the data attributes used to define the :class:`Mesh <blueprints.geoms.Mesh>`. 
	All vertex, face, normals and texture coordinate data are stored in the MeshAssets :class:`MeshCache <blueprints.cache.MeshCache>` 
	in the :attr:`cache` attribute.

	In general, all Mesh data that are structured for xml representation and insufficient for user 
	manipulation are named as singular :attr:`vertex`, :attr:`face`, :attr:`texcoord` and :attr:`normal` 
	while their plural form :attr:`vertecies`, :attr:`faces`, :attr:`cache.texcoords <blueprints.cache.MeshCache.texcoords>` and :attr:`cache.normals <blueprints.cache.MeshCache.normals>` are used 
	for user interpretable representations and manipulation.

	.. note::
		The position of MeshAssets is defined in mujoco via the :attr:`refpos` attribute, which differs 
		to the standard ``pos`` attribute in that it defines the reference frames relative position w.r.t. 
		to its position instead of the vice versa the relative position w.r.t. the reference frames 
		origin, effectively making ``refpos = - pos``. We find this to be unnecessarily confusing and 
		opted to implement the standard position attribute only translating when the kinematic tree gets 
		converted to Mujoco xml. It is highly recommended to use the :attr:`pos` argument to modify the 
		MeshAssets position.
	
	Attributes
	----------
	built : bool
		A flag indicating whether the MeshAssets has been built. This is necessary since multiple 
		:class:`Meshes <blueprints.geoms.Mesh>` might use the same :class:`MeshAsset` in which case 
		it should be built only once.
	cache : blue.CacheType
		The :class:`MeshCache <blueprints.cache.MeshCache>` which stores the Meshs data.
	filename : str
		The filename from which the Mesh data is loaded or to which it will be saved.
	xml_data : bool
		A flag indicating whether the Mesh data is written into the xml directly or into and external 
		file which is then linked to in the xml asset element. 
	"""
	
	@blue.restrict
	def __init__(self, 
		     pos:       np.ndarray|list[int|float] = [0., 0., 0.],
		     vertecies: np.ndarray|list[np.ndarray|list[float|int]]|None = None,  
		     faces:     list[np.ndarray|list[float|int]]|None            = None,
		     scale:     np.ndarray|list[int|float] = [1., 1., 1.], 
		     filename:  str|None            = None, 
		     centered:  bool                = False, 
		     xml_data:  bool                = False, 
		     cache:     blue.ThingType|None = None, 
		     name:      str|None            = None, 
		     **kwargs) -> None:
		"""
		Parameters
		----------
		pos : np.ndarray | list[int | float], optional
			pos : np.ndarray | list[int | float], optional
			Represents the position of the Asset. Changing this attribute also changes the properties 
			:attr:`x`, :attr:`y` and :attr:`z`.
		vertecies : np.ndarray | list[np.ndarray | list[float | int]] | None, optional
			A list of all vertex positions. The vertecies are stored in :attr:`cache`.
		faces : list[np.ndarray | list[float | int]] | None, optional
			A list of all faces. A face consists of at least 3 indecies of vertecies. The faces are 
			stored in :attr:`cache`.
		scale : np.ndarray | list[int | float], optional
			The scale is a 1D array like object of length 3. Each component is used to scale the X, 
			Y and Z-axis respectively. Changing the scale is computationally cheap since it does not 
			change the vertecies or other data in :attr:`cache`.
		filename : str | None, optional
			The filename from which the mesh data is loaded and to to which the mesh will be saved. 
			The file is saved in a special directory such that no input file is overwritten by the 
			output file.
		centered : bool, optional
			If ``True`` the vertecies are normalized such that their mean position is the reference 
			frames origin.
		xml_data : bool, optional
			If ``True`` the Mesh data is not written to an external file and linked to in the xml, 
			but written into the xml directly. It is recommended to only use this option if the Mesh 
			data is small enough to not blot the xml.
		cache : blue.ThingType | None, optional
			The :class:`Cache <blueprints.cache.MeshCache>` which stores the Mesh data.
		name : str | None, optional
			The user specified name might potentially be altered to avoid a naming conflict by 
			appending an enumeration scheme.
		**kwargs
			Keyword arguments are passed to ``super().__init__``.
		"""
		if cache is None:
			self.cache = blue.cache.MeshCache(vertecies=vertecies, 
							  faces=faces, 
							  filename=filename, 
							  centered=centered)
		else:
			self.cache = cache
		self.cache._add(self)
		self.xml_data = xml_data
		#self.filename = filename
		self.scale    = scale
		self._built    = False
		super().__init__(pos=pos, 
				 name=name, 
				 **kwargs)
		#blue.REGISTER.assets.append(self)
		if name is None:
			self._name = None


	@blue.restrict
	def load(self, filename: str) -> None:
		"""
		This methods loads the mesh data from the filename to the assets cache.
		
		Parameters
		----------
		filename : str
			The filename must be binary or ascii for OBJ files or binary for STL files.
		"""
		self.cache.load(filename)


	@blue.restrict
	def save(self, filename: str) -> None:
		"""
		This method saves the mesh data from the cache to the filename.
		
		Parameters
		----------
		filename : str
			The filename must either end with the ``'.obj`` or ``'.stl'`` file extension. Both file 
			formats are saved as binary files.
		"""
		self.cache.save(filename)


	@blue.restrict
	@classmethod
	def _from_xml_element(cls, 
			      xml_element: xml.Element, 
			      cache:       blue.CacheType) -> blue.ThingType:
		"""
		This method reconstructs an MeshAsset from an xml element.
		
		Parameters
		----------
		xml_element : xml.Element
			The xml element from which a MeshAsset is reconstructed.
		cache : blue.CacheType
			The Cache which stores the Mesh data.
		
		Returns
		-------
		blue.ThingType
			The reconstructed MeshAsset.
		"""
		init_args, post_args, rest_args = cls._xml_element_args(xml_element)
		init_args['cache'] = cache
		if 'refpos' in post_args:
			init_args['pos'] = - post_args.pop('refpos')
		if 'vertex' in rest_args:
			init_args['xml_data']  = True
			init_args['vertecies'] = rest_args.pop('vertex')
		else:
			init_args['xml_data']  = False
		mesh = object.__new__(cls)
		mesh.__init__(**init_args)
		for key, val in post_args.items():
			setattr(mesh, key, val)
		return mesh


	def _build(self, parent, world, indicies, **kwargs):
		"""
		This method is called by the Assets parent to construct the xml representations of 
		the kinematic tree.
		
		Parameters
		----------
		parent : xml.etree.ElementTree.Element
			The xml element of its parent
		world : WorldType
			The World from which the build method was called initially
		**kwargs
			Keyword arguments are overwrite the mujoco specs for building.
		
		Returns
		-------
		xml.etree.ElementTree.Element
			The built xml element of the Asset.
		"""
		if not self._built:
			if self.xml_data:
				mujoco_specs = self._mujoco_specs()
				mujoco_specs.update(self.cache._mujoco_specs())
				self._xml_root = xml.Element('mesh', **mujoco_specs)
				self.cache._ROOT = world
			else:
				# MAKE DIRECTORIES
				assets_path = f'{world._path}/{ASSET_DIR}'
				if not os.path.exists(assets_path):
					os.mkdir(assets_path)
				dirname = f'{world._path}/{ASSET_DIR}/meshes'
				if not os.path.exists(dirname):
					os.mkdir(dirname)
				# BUILD CACHE AND SELF
				self.cache._build(dirname)
				self._xml_root = xml.Element('mesh', 
							     file=self.cache._path, 
							     **self._mujoco_specs({'vertex': None, 'face': None}))
			self._index = indicies['mesh']
			self._built = True
		if self._xml_root not in world._xml_asset:
			world._xml_asset.append(self._xml_root)
		return self._xml_root

	# MUJOCO PROPERTIES

	@property
	def file(self) -> str | None:
		"""
		Derived dummy variable for Mujoco XML

		Returns
		-------
		str
		"""
		return self.filename


	@property
	def filename(self) -> str | None:
		"""
		The filename of the Mesh

		Returns
		-------
		str
		"""
		return self.cache.filename


	@filename.setter
	@blue.restrict
	def filename(self, filename: str | None) -> None:
		"""
		Returns
		-------
		str
		"""
		self.cache.filename = filename


	@property
	def refpos(self) -> np.ndarray:
		"""
		The negative :attr:`pos`. This attribute is implemented for mujoco, but synchronized with 
		:attr:`pos` which is recommended to to be used instead.
		
		Returns
		-------
		np.ndarray
		"""
		return - self.pos


	@property
	@blue.restrict
	def vertex(self) -> np.ndarray:
		"""
		This attribute is used to write construct the mujoco xml. It returns a flattened list of 
		coordinates that is written raw to xml. For a structured list of vertecies see :attr:`vertecies`.

		Returns
		-------
		np.ndarray
		"""
		return self.cache.vertex


	@property
	@blue.restrict
	def face(self) -> np.ndarray|None:
		"""
		This attribute is used to write to xml. It returns a flattened list of indecies that is 
		written raw to xml. For a structured list of faces see :attr:`faces`.

		Returns
		-------
		np.ndarray | None
		"""
		return self.cache.face


	@face.setter
	@blue.restrict
	def face(self, face: list) -> None:
		"""
		Parameters
		----------
		face : list
			This attribute is used to write to xml. It returns a flattened list of indecies that is 
			written raw to xml. For a structured list of faces see :attr:`faces`.
		"""
		self.cache._prepare_for_modification(self)
		self.cache.face = face


	@property
	def texcoord(self) -> np.ndarray|None:
		"""
		This attribute is used to write to xml. It returns a flattened list of texture coordinates that 
		is written raw to xml. For a structured list of texture coordinates see :attr:`texcoords`.
		
		Returns
		-------
		np.ndarray | None
		"""
		return self.cache.texcoord


	@texcoord.setter
	@blue.restrict
	def texcoord(self, texcoord: list) -> None:
		"""
		Parameters
		----------
		texcoord : list
			This attribute is used to write to xml. It returns a flattened list of texture 
			coordinates that is written raw to xml. For a structured list of texture coordinates see 
			:attr:`texcoords`.
		"""
		self.cache._prepare_for_modification(self)
		self.cache.texcoord = texcoord


	@property
	def normal(self) -> np.ndarray|None:
		"""
		This attribute is used to write to xml. It returns a flattened list of face normals that is 
		written raw to xml. For a structured list of face normals see :attr:`normals`.
		
		Returns
		-------
		np.ndarray | None
		"""
		return self.cache.normal


	@normal.setter
	@blue.restrict
	def normal(self, normal: list) -> None:
		"""
		Parameters
		----------
		normal : list
			This attribute is used to write to xml. It returns a flattened list of face normals that 
			is written raw to xml. For a structured list of face normals see :attr:`normals`.
		"""
		self.cache._prepare_for_modification(self)
		self.cache.normal = normal


	@property
	def size(self) -> np.ndarray:
		"""
		The size attribute specifies the ranges for each axis that the Mesh occupies.
		
		Returns
		-------
		np.ndarray
		"""
		return self.scale * self.cache.size


	@size.setter
	@blue.restrict
	def size(self, size: np.ndarray|list[int|float]) -> None:
		"""
		The size attribute specifies the ranges for each axis that the Mesh occupies.
		
		Parameters
		----------
		size : np.ndarray | list[int | float]
			Changing this parameter scales the vertecies directly in the :class:`Cache <blueprints.cache.MeshCache>` 
			which is computationally costly. To scale the size of the Mesh independently from the 
			vertecies in a computationally cheap manner use the :attr:`scale` attribute.
		"""
		self.cache._prepare_for_modification(self)
		size = np.array(size, dtype=np.float32) / self.scale
		size[np.isinf(size)] = 0
		self.cache.size = size


	@property
	def scale(self) -> np.ndarray:
		"""
		The scale is a 1D array like object of length 3. Each component is used to scale the X, Y and 
		Z-axis respectively. Changing the scale is computationally cheap since it does not change the 
		vertecies or other data in :attr:`cache`.
		
		Returns
		-------
		np.ndarray
		"""
		return self._scale


	@scale.setter
	@blue.restrict
	def scale(self, scale: np.ndarray|list[int|float]):
		"""
		Parameters
		----------
		scale : np.ndarray | list[int | float]
			The scale is a 1D array like object of length 3. Each component is used to scale the X, 
			Y and Z-axis respectively. Changing the scale is computationally cheap since it does not 
			change the vertecies or other data in :attr:`cache`.
		
		Raises
		------
		ValueError
			If the scale contains a component that is 0 an error is raised.
		"""
		scale = np.array(scale, dtype=np.float32)
		if np.any(scale == 0):
			raise ValueError(f'A meshes scale is not allowed to be zero in any component got {scale}.')
		self._scale = scale

	# BLUEPRINTS PROPERTIES

	@property
	def vertecies(self) -> list[np.ndarray]:
		"""
		The list of vertecies. Each vertex is a np.ndarray with 3 components for each of the 
		spatial dimensions.
		
		Returns
		-------
		list[np.ndarray]
		"""
		return self.cache.vertecies


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

		self.cache._prepare_for_modification(self)
		self.cache.vertecies = vertecies

	
	@property
	def faces(self) -> list[np.ndarray]|None:
		"""
		The list of faces. Each face is a np.ndarray with n indecies for each of the vertecies of the 
		face.
		
		Returns
		-------
		list[np.ndarray] | None
		"""
		return self.cache.faces


	@faces.setter
	@blue.restrict
	def faces(self, faces: np.ndarray|list[np.ndarray|list[float|int]]) -> None:
		"""
		Parameters
		----------
		faces : np.ndarray | list[np.ndarray | list[float | int]]
			The list of faces. Each face is a np.ndarray with n indecies for each of the vertecies 
			of the face.
		"""
		self.cache._prepare_for_modification(self)
		self.cache.faces = faces



class HFieldAsset(blue.HFieldAssetType, BaseAsset, blue.thing.MoveableThing):
	"""
	See :class:`HField <blueprints.geoms.HField>`
	
	Most attribute descriptions are partially taken from `Mujoco <https://mujoco.readthedocs.io/en/latest/XMLreference.html#body-geom>`__.
	"""
	@blue.restrict
	def __init__(self, 
		     filename: str|None                        = None, 
		     pos:      np.ndarray|list[int|float]      = [0., 0., 0.], 
		     terrain:  np.ndarray|list[int|float]|None = None, 
		     x_length: int|float|None                  = 1, 
		     y_length: int|float|None                  = 1, 
		     z_length: int|float|None                  = 1, 
		     height_offset: int|float|None             = 1, 
		     xml_data: bool                            = False, 
		     cache:    blue.ThingType|None             = None, 
		     name:     str|None                        = None, 
		     **kwargs) -> None:
		if cache is None:
			self.cache = blue.cache.HFieldCache(terrain=terrain, 
							    filename=filename)
		else:
			self.cache = cache
		self.cache._add(self)
		self.xml_data = xml_data
		self.filename = filename
		self.x_length = x_length
		self.y_length = y_length
		self.z_length = z_length
		self.height_offset = height_offset
		self._built    = False
		super().__init__(pos=pos, 
				 name=name, 
				 **kwargs)
		if name is None:
			self._name = None


	@blue.restrict
	def _build(self, parent, world, indicies, **kwargs):
		"""
		This method is called by the Assets parent to construct the xml representations of 
		the kinematic tree.
		
		Parameters
		----------
		parent : xml.etree.ElementTree.Element
			The xml element of its parent
		world : WorldType
			The World from which the build method was called initially
		**kwargs
			Keyword arguments are overwrite the mujoco specs for building.
		
		Returns
		-------
		xml.etree.ElementTree.Element
			The built xml element of the Asset.
		"""
		if not self._built:
			if self.xml_data and False:
				mujoco_specs = self._mujoco_specs()
				mujoco_specs.update(self.cache._mujoco_specs())
				self._xml_root = xml.Element('hfield', **mujoco_specs)
			else:
				# MAKE DIRECTORIES
				if not os.path.exists(ASSET_DIR):
					os.mkdir(ASSET_DIR)
				assets_path = f'{world._path}/{ASSET_DIR}/hfields'
				if not os.path.exists(assets_path):
					os.mkdir(assets_path)
				# BUILD CACHE AND SELF
				self.cache._build(assets_path)
				self._xml_root = xml.Element('hfield', 
							      file=self.cache._path, 
							      **self._mujoco_specs({'elevation': None}))
			self.cache._world   = world
			self.cache._index   = indicies['hfield']
			self._built = True
		if self._xml_root not in world._xml_asset:
			world._xml_asset.append(self._xml_root)
		self.cache._ROOT = world
		return self._xml_root


	def __getitem__(self, 
			key: tuple[slice|int]) -> np.ndarray|np.float32:
		"""
		Returns
		-------
		np.ndarray | np.float32
			The values from the selected index/slice of the height field.
		"""
		return self.cache[key]


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
		self.cache._prepare_for_modification(self)
		self.cache[key] = value

	# MUJOCO PROPERTIES

	@property
	def size(self) -> np.ndarray:
		"""
		Derived size attribute for Mujoco XML.

		Returns
		-------
		np.ndarray
		"""
		return np.array([self.x_length/2, self.y_length/2, self.z_length/2, self.height_offset/2])


	@property
	def name(self) -> str:
		"""
		The user specified name might potentially be altered to avoid a naming conflict by appending an 
		enumeration scheme.
		
		Returns
		-------
		str
		"""
		return self._name or f'asset_{self.ID}'


	@name.setter
	@blue.restrict
	def name(self, name: str) -> None:
		"""
		The user specified name might potentially be altered to avoid a naming conflict by appending an 
		enumeration scheme.
		
		Parameters
		----------
		name : str
			The user specified name.
		"""
		self._name = name

	# BLUEPRINTS PROPERTIES
		
	@property
	def x_length(self):
		"""
		Length along the ``X``-axis of the HField.

		Returns
		-------
		float
		"""
		return self._x_length


	@property
	def y_length(self):
		"""
		Length along the ``Y``-axis of the HField.

		Returns
		-------
		float
		"""
		return self._y_length


	@property
	def z_length(self):
		"""
		Length along the ``Z``-axis of the HField.

		Returns
		-------
		float
		"""
		return self._z_length


	@property
	def height_offset(self):
		"""
		Height offset along the ``Z``-axis of the HField.

		Returns
		-------
		float
		"""
		return self._height_offset


	@x_length.setter
	@blue.restrict
	def x_length(self, x_length: int|float|np.int32|np.int64|np.float32|np.float64) -> None:
		self._x_length = float(x_length)


	@y_length.setter
	@blue.restrict
	def y_length(self, y_length: int|float|np.int32|np.int64|np.float32|np.float64) -> None:
		self._y_length = float(y_length)


	@z_length.setter
	@blue.restrict
	def z_length(self, z_length: int|float|np.int32|np.int64|np.float32|np.float64) -> None:
		self._z_length = float(z_length)


	@height_offset.setter
	@blue.restrict
	def height_offset(self, height_offset: int|float|np.int32|np.int64|np.float32|np.float64) -> None:
		self._height_offset = float(height_offset)
	

	@property
	def terrain(self) -> list[np.ndarray]:
		"""
		The terrain data of the HField
		
		Returns
		-------
		list[np.ndarray]
		"""
		return self.cache.terrain


	@terrain.setter
	@blue.restrict
	def terrain(self, terrain: np.ndarray|list[np.ndarray|list[float|int]]) -> None:
		"""
		Parameters
		----------
		vertecies : np.ndarray | list[np.ndarray | list[float | int]]
		"""
		self.cache._prepare_for_modification(self)
		self.cache.terrain = terrain



ASSET_THINGS  = {'mesh':   MeshAsset, 
		 'hfield': HFieldAsset}
