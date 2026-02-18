"""
There are currently three types of textures supported in blueptints.
Boxes, 2D-Planes and Meshtextures. To apply a texture to a Thing it has 
to be mediated through a :class:`Material <blueprints.material.Material>`.

.. code-block::
	:caption: Texture asssignment

	>>> texture  = blue.texture.Plane()
	>>> material = blue.Material(texture=texture)
	>>> plane    = blue.geoms.Plane(material=material)


Texture Types
=============
We will present the basic Texture Types here for image textures. The 
same types are also usable for Procedural Textures (see below).

Plane
-----

The :class:`Plane` texture encodes 2D images which can be provided by a file path. 
For clarity the MIME type and the number of channels can be made explicit with 
``content`` and ``n_channel``. To flip the image vertically or horizontally 
``v_flip`` and ``h_flip`` can be used.

.. code-block::
	:caption: Plane Texture

	>>> sand_tex = blue.texture.Plane(filename='sand.png')
	>>> sand_mat = blue.Material(texture=sand_tex)
	>>> blue.geoms.Plane(material=sand_mat)

.. image:: /_static/texture_plane.png

Box
---

There are two ways of specifying the six sides of a :class:`Box` we give the first 
as an example here and the second for the equivalent :class:`Skybox` texture.
The provided file paths are as always taken relative to the file instantiating the World.

.. code-block::
	:caption: Box Texture

	>>> grass_tex = blue.texture.Box(filename_up   ='grass_top.png', 
	>>> 				 filename_left ='grass_side.png', 
	>>> 				 filename_right='grass_side.png', 
	>>> 				 filename_front='grass_side.png', 
	>>> 				 filename_back ='grass_side.png', 
	>>> 				 filename_down ='grass_bottom.png')
	>>> grass_mat = blue.Material(name='grass', texture=grass_tex)
	>>> grass = blue.geoms.Box(material=grass_mat)

.. image:: /_static/texture_box.png


Mesh
----

If a :class:`Mesh <blueprints.geoms.Mesh>` has texture coordinates (like ``.obj`` files) 
a :class:'Plane' texture can be applied to it. If a mesh does not have texture coordinates 
they can also be provided manually from the user through :attr:`Mesh.texcoord <blueprints.assets.MeshAsset.texcoord>`.

.. code-block::
	:caption: Mesh Texture

	>> creeper_tex = blue.texture.Plane(filename='creeper.png')
	>> creeper_mat = blue.Material(texture=creeper_tex)
	>> creeper = blue.geoms.Mesh(filename='creeper.obj')

.. image:: /_static/texture_mesh.png

SkyBox
------

Instead of applying a :class:`Skybox` texture via a :class:`Material <blueprints.material.Material>` 
it can be applied directly to the :class:`World <blueprints.world.World>` through the :attr:`World.texture <blueprints.world.World.texture>` 
attribute. Skybox and Box textures use equvalent attributes and constructions.

.. warning::
	As of Mujoco 3.3.2 there is a bug that prevents a skybox from being loaded if no other textured :class:`Material <blueprints.material.Material>` 
	if included in the :class:`World <blueprints.world.World>`.

To load a cube like texture from a single file it is partitioned into a grid through the 
``grid_size`` argument. The number of cells (i.e. the product of ``grid_size``) must not 
exceed 12. The assignment of each cell to a side of the cube is then specified via the 
``grid_layout`` string. Each row is read left to right with the character,
``'U'`` up, ``'D'`` down, ``'L'`` left, ``'R'`` right, ``'F'`` front, ``'B'`` back and ``'.'`` skip 
mapping sides to grid cells.

.. image:: /_static/bluesky_grid.png

.. code-block::
	:caption: Skybox Texture

	>>> sky = blue.texture.Skybox(filename='sky.png', 
	>>> 			      grid_layout='.U..LFRB.D..', 
	>>> 			      grid_size=[3, 4])
	>>> world.texture = sky

.. image:: /_static/texture_sky.png


Procedural Textures
===================
Mujoco supports simple procedurally generated textures which are accessable 
through the bultin attribute. Their apperance for :class:`Box Textures <blueprints.texture.Box>` 
and :class:`Plane Textures <blueprints.texture.Plane>` differs so we present both in the 
following examples.

.. note::
	Though they are named identically and we present Box Geoms with Box Textures 
	and Plane geoms with Plane Textures, the opposite assignment is also possible.


.. code-block::
	:caption: Example

	>>> box_tex = ...
	>>> plane_tex = ...
	>>> ...
	>>> box_mat = blue.Material(texture=box_tex)
	>>> box = blue.geoms.Box(material=box_mat, 
	>>> 			 x=-1.5,
	>>> 			 z=0.5, 
	>>> 			 gamma=TAU/8)
	>>> plane_mat = blue.Material(texture=plane_tex, 
	>>> 			      texrepeat=[2, 1])
	>>> plane = blue.geoms.Plane(material=plane_mat, 
	>>> 			     x_length=2, 
	>>> 			     y_length=1,
	>>> 			     x=1.5, 
	>>> 			     z=0.5)
	>>> world.attach(box, plane, blue.geoms.Plane(color='orange'))


Gradient
--------

.. code-block::
	:caption: Gradient Builtin Texture

	>>> box_tex = blue.texture.Box(builtin='gradient', 
	>>> 			       color_1='white', 
	>>> 			       color_2='black', 
	>>> 			       width=1000, 
	>>> 			       height=1000)
	>>> plane_tex = blue.texture.Plane(builtin='gradient',
	>>> 				   color_1='white', 
	>>> 				   color_2='black', 
	>>> 				   width=1000, 
	>>> 				   height=1000)
	>>> ...


.. image:: /_static/texture_gradient.png


Checker
-------

.. code-block::
	:caption: Checker Builtin Texture

	>>> box_tex = blue.texture.Box(builtin='checker', 
	>>> 			       color_1='white', 
	>>> 			       color_2='black', 
	>>> 			       width=1000, 
	>>> 			       height=1000)
	>>> plane_tex = blue.texture.Plane(builtin='checker',
	>>> 				   color_1='white', 
	>>> 				   color_2='black', 
	>>> 				   width=1000, 
	>>> 				   height=1000)
	>>> ...


.. image:: /_static/texture_checker.png

Flat
----

.. code-block::
	:caption: Flat Builtin Texture

	>>> box_tex = blue.texture.Box(builtin='flat', 
	>>> 			       color_1='white', 
	>>> 			       color_2='black', 
	>>> 			       width=1000, 
	>>> 			       height=1000)
	>>> plane_tex = blue.texture.Plane(builtin='flat',
	>>> 				p   color_1='white', 
	>>> 				   color_2='black', 
	>>> 				   width=1000, 
	>>> 				   height=1000)
	>>> ...


.. image:: /_static/texture_flat.png


Mark
----

Additionally there are three variations of markings: ``'edge'``, ``'cross'`` and ``'random'``.
The last marking takes the additional argument ``random`` indicating the probability of a texture 
pixel being colored with the ``color_mark``.

.. code-block::
	:caption: Mark Texture

	>>> edge_tex = blue.texture.Box(builtin='flat', 
	>>> 				color_1='white', 
	>>> 				color_2='black', 
	>>> 				color_mark='orange', 
	>>> 				mark='edge', 
	>>> 				width=50, 
	>>> 				height=50)
	>>> cross_tex = blue.texture.Box(builtin='flat', 
	>>> 				 color_1='white', 
	>>> 				 color_2='black', 
	>>> 				 color_mark='orange', 
	>>> 				 mark='cross', 
	>>> 				 width=50, 
	>>> 				 height=50)
	>>> random_1_tex = blue.texture.Box(builtin='flat', 
	>>> 				    color_1='white', 
	>>> 				    color_2='black', 
	>>> 				    color_mark='orange', 
	>>> 				    mark='random', 
	>>> 				    random=0.05,
	>>> 				    width=50, 
	>>> 				    height=50)
	>>> random_2_tex = blue.texture.Box(builtin='flat', 
	>>> 				    color_1='white', 
	>>> 				    color_2='black', 
	>>> 				    color_mark='orange', 
	>>> 				    mark='random', 
	>>> 				    random=0.5,
	>>> 				    width=50, 
	>>> 				    height=50)


.. image:: /_static/texture_mark.png

"""


import numpy as np
import blueprints as blue
from blueprints.thing.colored import Color



class BaseTexture(blue.BaseThing):
	"""
	Most attribute descriptions are partially taken from `Mujoco <https://mujoco.readthedocs.io/en/latest/XMLreference.html#asset-texture>`__.
	"""
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
		self.asset._build(parent=parent, 
				  world=world, 
				  indicies=indicies, 
				  **kwargs)
		return self.asset._xml_root


	@property
	def asset(self) -> blue.TextureAssetType:
		"""
		The attributes of the texture are stored in a separate asset object.
		If this texture is assign to multiple objects, and then gets modified 
		through ``Material.texture.attribute = value`` all  other Materials 
		referenceing the Texture will be unaltered. See :class:`UniqueThing <blueprints.thing.unique.UniqueThing>`.

		Returns
		-------
		blue.TextureAssetType
		"""
		return self._asset


	@asset.setter
	@blue.restrict
	def asset(self, asset: blue.TextureAssetType) -> None:
		self._asset = asset	

	# BLUEPRINT PROPERTIES

	@property
	def content(self, content) -> str|None:
		"""
		Content type (MIME type) taking values ``None``, ``'image/png'``, 
		``'image/ktx'`` and ``'image/vnd.mujoco.texture'``.

		Returns
		-------
		str | None
		"""
		return self.asset.content


	@content.setter
	@blue.restrict
	def content(self, content: str|None) -> None:
		self.asset._prepare_for_modification(self)
		self.asset.content = content


	@property
	def filename(self, file) -> str|None:
		"""
		Name of the texture file.

		Returns
		-------
		str | None
		"""
		return self.asset.file


	@filename.setter
	@blue.restrict
	def filename(self, filename: str|None) -> None:
		self.asset._prepare_for_modification(self)
		self.asset.file = filename


	@property
	def grid_size(self, grid_size) -> list[int]:
		"""
		Grid size attrigute ``[n_rows, n_cols]`` defining the grid. 
		For non-grid Textures this value is ignored.

		Returns
		-------
		list[int]
		"""
		return self.asset.grid_size


	@grid_size.setter
	@blue.restrict
	def grid_size(self, grid_size: list[int]) -> None:
		self.asset._prepare_for_modification(self)
		self.asset.grid_size = grid_size


	@property
	def grid_layout(self, grid_layout) -> str|None:
		"""
		The layout of the grid. See example above.

		Returns
		-------
		str | None
		"""
		return self.asset.grid_layout


	@grid_layout.setter
	@blue.restrict
	def grid_layout(self, grid_layout: str|None) -> None:
		self.asset._prepare_for_modification(self)
		self.asset.grid_layout = grid_layout


	@property
	def filename_right(self, filename_right) -> str|None:
		"""
		The file name for the right side of a :class:`Box`.

		Returns
		-------
		str | None
		"""
		return self.asset.filename_right


	@filename_right.setter
	@blue.restrict
	def filename_right(self, filename_right: str|None) -> None:
		self.asset._prepare_for_modification(self)
		self.asset.filename_right = filename_right


	@property
	def filename_left(self, filename_left) -> str|None:
		"""
		The file name for the left side of a :class:`Box`.
		
		Returns
		-------
		str | None
		"""
		return self.asset.filename_left


	@filename_left.setter
	@blue.restrict
	def filename_left(self, filename_left: str|None) -> None:
		self.asset._prepare_for_modification(self)
		self.asset.filename_left = filename_left


	@property
	def filename_up(self, filename_up) -> str|None:
		"""
		The file name for the upper side of a :class:`Box`.
		
		Returns
		-------
		str | None
		"""
		return self.asset.filename_up


	@filename_up.setter
	@blue.restrict
	def filename_up(self, filename_up: str|None) -> None:
		self.asset._prepare_for_modification(self)
		self.asset.filename_up = filename_up


	@property
	def filename_down(self, filename_down) -> str|None:
		"""
		The file name for the down side of a :class:`Box`.
		
		Returns
		-------
		str | None
		"""
		return self.asset.filename_down


	@filename_down.setter
	@blue.restrict
	def filename_down(self, filename_down: str|None) -> None:
		self.asset._prepare_for_modification(self)
		self.asset.filename_down = filename_down


	@property
	def filename_front(self, filename_front) -> str|None:
		"""
		The file name for the front side of a :class:`Box`.
		
		Returns
		-------
		str | None
		"""
		return self.asset.filename_front


	@filename_front.setter
	@blue.restrict
	def filename_front(self, filename_front: str|None) -> None:
		self.asset._prepare_for_modification(self)
		self.asset.filename_front = filename_front


	@property
	def filename_back(self, filename_back) -> str|None:
		"""
		The file name for the back side of a :class:`Box`.
		
		Returns
		-------
		str | None
		"""
		return self.asset.filename_back


	@filename_back.setter
	@blue.restrict
	def filename_back(self, filename_back: str|None) -> None:
		self.asset._prepare_for_modification(self)
		self.asset.filename_back = filename_back


	@property
	def builtin(self, builtin) -> str|None:
		"""
		If set, this attribute specifies a builtin procedural Texture. 
		See examples above.

		Returns
		-------
		str | None
		"""
		return self.asset.builtin


	@builtin.setter
	@blue.restrict
	def builtin(self, builtin: str|None) -> None:
		self.asset._prepare_for_modification(self)
		self.asset.builtin = builtin


	@property
	def color_1(self, color_1) -> blue.ColorType:
		"""
		The first color of a procedural Texture.

		Returns
		-------
		blue.ColorType
		"""
		return self.asset.color_1


	@color_1.setter
	@blue.restrict
	def color_1(self, color_1: object) -> None:
		self.asset._prepare_for_modification(self)
		self.asset.color_1 = color_1


	@property
	def color_2(self, color_2) -> blue.ColorType:
		"""
		The second color of a procedural Texture.
		
		Returns
		-------
		blue.ColorType
		"""
		return self.asset.color_2


	@color_2.setter
	@blue.restrict
	def color_2(self, color_2: object) -> None:
		self.asset._prepare_for_modification(self)
		self.asset.color_2 = color_2


	@property
	def mark(self, mark) -> str|None:
		"""
		If set this attribute specifies additional markings on 
		a procedural Texture. Possible values are ``None``, 
		``'edge'``, ``'cross'`` and ``'random'``. See examples above.
		
		Returns
		-------
		str | None
		"""
		return self.asset.mark


	@mark.setter
	@blue.restrict
	def mark(self, mark: str|None) -> None:
		self.asset._prepare_for_modification(self)
		self.asset.mark = mark


	@property
	def color_mark(self, color_mark) -> blue.ColorType:
		"""
		The mark color of a procedural Texture.
		
		Returns
		-------
		blue.ColorType
		"""
		return self.asset.color_mark


	@color_mark.setter
	@blue.restrict
	def color_mark(self, color_mark: object) -> None:
		self.asset._prepare_for_modification(self)
		self.asset.color_mark = color_mark


	@property
	def random(self, random) -> float:
		"""
		If the ``mark`` attribute is set to ``'random'``, this 
		attribute specifies the probability of a pixel in the 
		texture being the ``mark_color``.

		Returns
		-------
		float
		"""
		return self.asset.random


	@random.setter
	@blue.restrict
	def random(self, random: float) -> None:
		self.asset._prepare_for_modification(self)
		self.asset.random = random


	@property
	def width(self, width) -> int:
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
		return self.asset.width


	@width.setter
	@blue.restrict
	def width(self, width: int) -> None:
		self.asset._prepare_for_modification(self)
		self.asset.width = width


	@property
	def height(self, height) -> int:
		"""
		The height of the procedural texture, i.e., the number of rows 
		in the image. For :class:`Box` and :class:`Skybox` textures, this attribute is 
		ignored and the height is set to 6 times the width. For textures 
		loaded from files, this attribute is ignored.

		Returns
		-------
		int
		"""
		return self.asset.height


	@height.setter
	@blue.restrict
	def height(self, height: int) -> None:
		self.asset._prepare_for_modification(self)
		self.asset.height = height


	@property
	def h_flip(self, h_flip) -> bool:
		"""
		If true, images loaded from file are flipped in the horizontal 
		direction. Does not affect procedural textures.

		Returns
		-------
		bool
		"""
		return self.asset.h_flip


	@h_flip.setter
	@blue.restrict
	def h_flip(self, h_flip: bool) -> None:
		self.asset._prepare_for_modification(self)
		self.asset.h_flip = h_flip


	@property
	def v_flip(self, v_flip) -> bool:
		"""
		If true, images loaded from file are flipped in the vertical 
		direction. Does not affect procedural textures.

		Returns
		-------
		bool
		"""
		return self.asset.v_flip


	@v_flip.setter
	@blue.restrict
	def v_flip(self, v_flip: bool) -> None:
		self.asset._prepare_for_modification(self)
		self.asset.v_flip = v_flip


	@property
	def n_channel(self, n_channel) -> int:
		"""
		The number of channels in the texture image file. This allows 
		loading 4-channel textures (RGBA) or single-channel textures 
		(e.g., for Physics-Based Rendering properties such as roughness 
		or metallic).

		Returns
		-------
		int
		"""
		return self.asset.n_channel


	@n_channel.setter
	@blue.restrict
	def n_channel(self, n_channel: int) -> None:
		self.asset._prepare_for_modification(self)
		self.asset.n_channel = n_channel



class Plane(blue.PlaneTextureType, BaseTexture):
	@blue.restrict
	def __init__(self, 
		     content:     str|None  =  None, 
		     filename:    str|None  =  None, 
		     builtin:     str|None  =  None, 
		     color_1:     object    = [0.8, 0.8, 0.8], 
		     color_2:     object    = [0.5, 0.5, 0.5], 
		     mark:        str|None  =  None, 
		     color_mark:  object    = [0.0, 0.0, 0.0],
		     random:      float     =  0.01, 
		     width:       int       =  1000, 
		     height:      int       =  1000, 
		     h_flip:      bool      =  False, 
		     v_flip:      bool      =  False, 
		     n_channel:   int       =  3, 
		     name:        str|None  = None, 
		     asset:       blue.TextureAssetType|None = None):
		if asset is None:
			self._asset = blue.assets.TextureAsset(content=content, 
							       file=filename, 
							       builtin=builtin, 
							       color_1=color_1, 
							       color_2=color_2, 
							       mark=mark, 
							       color_mark=color_mark, 
							       random=random, 
							       width=width, 
							       height=height, 
							       h_flip=h_flip, 
							       v_flip=v_flip, 
							       n_channel=n_channel, 
							       name=name)
		else:
			self._asset = asset
		self.asset._add(self)
		self.asset._TYPE = self._TYPE
		super().__init__(name=name)



class Box(blue.BoxTextureType, BaseTexture):
	@blue.restrict
	def __init__(self, 
		     content:         str|None  =  None, 
		     filename:        str|None  =  None, 
		     grid_size:       list[int] = [1, 1], 
		     grid_layout:     str|None  =  None,
		     filename_right:  str|None  =  None, 
		     filename_left:   str|None  =  None, 
		     filename_up:     str|None  =  None, 
		     filename_down:   str|None  =  None, 
		     filename_front:  str|None  =  None, 
		     filename_back:   str|None  =  None, 
		     builtin:         str|None  =  None, 
		     color_1:         object    = [0.8, 0.8, 0.8], 
		     color_2:         object    = [0.5, 0.5, 0.5], 
		     mark:            str|None  =  None, 
		     color_mark:      object    = [0.0, 0.0, 0.0],
		     random:          float     =  0.01, 
		     width:           int       =  1000, 
		     height:          int       =  1000, 
		     h_flip:          bool      =  False, 
		     v_flip:          bool      =  False, 
		     n_channel:       int       =  3, 
		     name:            str|None  = None, 
		     asset:           blue.TextureAssetType|None = None):
		if asset is None:
			self._asset = blue.assets.TextureAsset(content=content, 
							       file=filename, 
							       grid_size=grid_size, 
							       grid_layout=grid_layout, 
							       fileright=filename_right, 
							       fileleft=filename_left, 
							       fileup=filename_up, 
							       filedown=filename_down, 
							       filefront=filename_front, 
							       fileback=filename_back, 
							       builtin=builtin, 
							       color_1=color_1, 
							       color_2=color_2, 
							       mark=mark, 
							       color_mark=color_mark, 
							       random=random, 
							       width=width, 
							       height=height, 
							       h_flip=h_flip, 
							       v_flip=v_flip, 
							       n_channel=n_channel, 
							       name=name)
		else:
			self._asset = asset
		self.asset._add(self)
		self.asset._TYPE = self._TYPE
		super().__init__(name=name)



class Skybox(blue.SkyboxTextureType, Box):
	@blue.restrict
	def __init__(self, 
		     content:         str|None  =  None, 
		     filename:        str|None  =  None, 
		     grid_size:       list[int] = [1, 1], 
		     grid_layout:     str|None  =  None,
		     filename_right:  str|None  =  None, 
		     filename_left:   str|None  =  None, 
		     filename_up:     str|None  =  None, 
		     filename_down:   str|None  =  None, 
		     filename_front:  str|None  =  None, 
		     filename_back:   str|None  =  None, 
		     builtin:         str|None  =  None, 
		     color_1:         object    = [0.8, 0.8, 0.8], 
		     color_2:         object    = [0.5, 0.5, 0.5], 
		     mark:            str|None  =  None, 
		     color_mark:      object    = [0.0, 0.0, 0.0],
		     random:          float     =  0.01, 
		     width:           int       =  1000, 
		     height:          int       =  1000, 
		     h_flip:          bool      =  False, 
		     v_flip:          bool      =  False, 
		     n_channel:       int       =  3, 
		     name:            str|None  = None, 
		     asset:           blue.TextureAssetType|None = None):
		super().__init__(content=content, 
				 filename=filename, 
				 grid_size=grid_size, 
				 grid_layout=grid_layout, 
				 filename_right=filename_right, 
				 filename_left=filename_left, 
				 filename_up=filename_up, 
				 filename_down=filename_down, 
				 filename_front=filename_front, 
				 filename_back=filename_back, 
				 builtin=builtin, 
				 color_1=color_1, 
				 color_2=color_2, 
				 mark=mark, 
				 color_mark=color_mark, 
				 random=random, 
				 width=width, 
				 height=height, 
				 h_flip=h_flip, 
				 v_flip=v_flip, 
				 n_channel=n_channel, 
				 name=name, 
				 asset=asset)
