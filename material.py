import numpy as np
import blueprints as blue



class Material(blue.MaterialType, blue.NodeThing):
	"""
	Materials can be given to visual Things to specify their reflective properties for the renderer. 
	To apply :class:`Textures <blueprints.texture.BaseTexture>` to Things, they must be attributes 
	of a Material which in turn is assigned to the Thing. Just keeping the default values for 
	Materials and assigning it to :class:`Sites <blueprints.sites.BaseSite>` and 
	:class:`Geoms <blueprints.geoms.BaseGeom>` will have no visual effect.
	
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
		     reflectance: int|float = 0., 
		     metallic:    int|float =-1.0, 
		     roughness:   int|float =-1.0, 
		     color:       str|object|None = None, 
		     name:        str|None        = None, 
		     asset:       blue.MaterialAssetType|None = None, 
		     copy:        bool = False):
		"""
		Parameters
		----------
		texture : blue.TextureType | None, optional
			A :class:`Texture <blueprints.texture.BaseTexture>` Thing
		texrepeat : list [ int | float ] | np.ndarray, optional
			The number of repetitions of the Texture in two directions
		texuniform : bool, optional
			For :class:`BoxTextures <blueprints.texture.Box>`, this attribute controls how 
			cube mapping is applied.
		emission : int | float, optional
			If set positive, the Material emitts lights.
		specular : int | float, optional
			Specularity in OpenGL. This value should be in the range [0 1].
		shininess : int | float, optional
			Shininess in OpenGL
		reflectance : int | float, optional
			Reflectance is computed without ray-tracing, so only a single reflection level 
			is rendered.
		metallic : int | float, optional
			This attribute is ignored by the default mujoco renderer.
		roughness : int | float, optional
			This attribute is ignored by the default mujoco renderer.
		color : str|object | None, optional
			The color of the Material (if set) will be multiplied with the 
			Things color and (if set) the Materials Texture.
		name : str | None, optional
			Name of the Material
		asset : blue.MaterialAssetType | None, optional
			The :class:`Asset <blueprints.asset.MaterialAsset>` of the Material
		copy : bool, optional
			Dummy argument without effect
		"""
		self._CHILDREN = dict() # Layers might be implemented someday
		if asset is None:
			self._asset = blue.assets.MaterialAsset(texture=texture, 
								texrepeat=texrepeat, 
								texuniform=texuniform, 
								emission=emission, 
								specular=specular, 
								shininess=shininess, 
								reflectance=reflectance, 
								metallic=metallic, 
								roughness=roughness, 
								color=color)
		else:
			self._asset = asset
		self.asset._add(self)
		super().__init__(name=name, color=color)


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
		self._asset._build(parent=parent, 
				   world=world, 
				   indicies=indicies, 
				   **kwargs)
		return self._asset._xml_root


	@property
	def asset(self) -> blue.MaterialAssetType:
		"""
		The :class:`Asset <blueprints.asset.MaterialAsset>` of the Material

		Returns
		-------
		blue.MaterialAssetType
		"""
		return self._asset


	@asset.setter
	@blue.restrict
	def asset(self, asset: blue.MaterialAssetType) -> None:
		#self._asset._remove(self)
		self._asset = asset
		#self._asset._add(self)
	


	@property
	def texrepeat(self) -> np.ndarray:
		"""
		The number of repetitions of the Texture in two directions

		Returns
		-------
		np.ndarray
		"""
		return self._asset.texrepeat


	@texrepeat.setter
	@blue.restrict
	def texrepeat(self, texrepeat: list[int|float]|np.ndarray) -> None:
		self.asset._prepare_for_modification(self)
		self.asset.texrepeat = texrepeat


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
		return self.asset.texuniform


	@texuniform.setter
	@blue.restrict
	def texuniform(self, texuniform: bool) -> None:
		self.asset._prepare_for_modification(self)
		self.asset.texuniform = texuniform


	@property
	def emission(self) -> float:
		"""
		If set positive, the Material emitts lights.
		
		Returns
		-------
		float
		"""
		return self.asset.emission


	@emission.setter
	@blue.restrict
	def emission(self, emission: int|float) -> None:
		self.asset._prepare_for_modification(self)
		self.asset.emission = emission


	@property
	def specular(self) -> float:
		"""
		Specularity in OpenGL. This value should be in the range [0 1].

		Returns
		-------
		float
		"""
		return self.asset.specular


	@specular.setter
	@blue.restrict
	def specular(self, specular: int|float) -> None:
		self.asset._prepare_for_modification(self)
		self.asset.specular = specular


	@property
	def shininess(self) -> float:
		"""
		Shininess in OpenGL.

		Returns
		-------
		float
		"""
		return self.asset.shininess


	@shininess.setter
	@blue.restrict
	def shininess(self, shininess: int|float) -> None:
		self.asset._prepare_for_modification(self)
		self.asset.shininess = shininess


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
		return self.asset.reflectance


	@reflectance.setter
	@blue.restrict
	def reflectance(self, reflectance: int|float) -> None:
		self.asset._prepare_for_modification(self)
		self.asset.reflectance = reflectance


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
		return self.asset.metallic


	@metallic.setter
	@blue.restrict
	def metallic(self, metallic: int|float) -> None:
		self.asset._prepare_for_modification(self)
		self.asset.metallic = metallic


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
		return self.asset.roughness


	@roughness.setter
	@blue.restrict
	def roughness(self, roughness: int|float) -> None:
		self.asset._prepare_for_modification(self)
		self.asset.roughness = roughness


	@property
	def color(self) -> blue.ColorType:
		"""
		The color of the Material if set will be multiplied 
		by the Things Color and Texture.

		Returns
		-------
		blue.ThingType
		"""
		return self.asset.color


	@color.setter
	@blue.restrict
	def color(self, color: str|object|None) -> None:
		#print(self.ID, self.asset._references)
		#print(self in self.asset._references)
		self.asset._prepare_for_modification(self)
		self.asset.color = color


	@property
	def texture(self) -> blue.TextureType:
		"""
		Referencing the material from a Thing will cause the 
		texture to be applied to that Thing.

		Returns
		-------
		blue.TextureType
		"""
		return self.asset.texture


	@texture.setter
	@blue.restrict
	def texture(self, texture: blue.TextureType) -> None:
		self.asset._prepare_for_modification(self)
		self.asset.texture = texture#.copy()
