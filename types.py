"""
.. note::
	If you found yourself here not being a dev, you should probably :mod:`skip this section <blueprints.world>` from or 
	look up the correspinging docs page of the Thing you were looking after for which this module only 
	implements its corresponding mirror types. A link to the docs page is to be found in the Types doc 
	string.

This module is used to model the inheritance hierarchy and to define the different attribute types.
Since blueprint specifies many values of different data types, the user will easily find themselves 
setting attributes to the wrong types. But for most attribute setters this would not be caught 
resulting in a downstream exception during the XML build. Those exceptions are not easily associated 
to a line in the users code, which violated a type constraint.

To prevent this, blueprints is type safe in attributes (which is most of the user accessable interface). 
The module typechecker is used to @restrict methods to only take arguments which satisfy the type 
of the method signature. In python those type hints are not binding, @restrict however will raise 
ValueError if a type hint is violated.

To make all types available during class definition for @restrict, the types module defines a mirror 
type for all Things. Those mirror types are to be used in the type hints for @restrict instead of the 
proper Thing class.
Every Thing class in blueprints must inherit (as its first base class) from a Type class in this module.
Type classes must mimic the inheritance of its corresponding class. This is used for type hints as
well as to ensure that the inheritance structure is preserved, when ``isinstance`` or ``issubclass``
is used.

Additionally the mirror types are used to specify class properties like data type associations for 
attributes used to translate back and forth between mujoco xml and blueprints.
"""


from abc import ABC, ABCMeta
import numpy as np



class PathType(ABC):

	"""
	This abstract base class is a used to represent the Tendon class in type hints. For a detailed
	description see :class:`blueprints.tendon.Path`.
	"""


class FunctionHandleType(ABC):

	"""
	This abstract base class is a used to represent the FunctionHandle class in type hints. For a detailed
	description see :class:`blueprints.utils.view.FunctionHandle`.
	"""


class ViewType(ABC):

	"""
	This abstract base class is a used to represent the View class in type hints. For a detailed
	description see :class:`blueprints.utils.view.View`.
	"""


class AllViewType(ViewType):

	"""
	This abstract base class is a used to represent the AllView class in type hints. For a detailed
	description see :class:`blueprints.utils.view.AllView`.
	"""


class LatticeType(ABC):

	"""
	This abstract base class is a used to represent the Lattice class in type hints. For a detailed
	description see :class:`blueprints.utils.lattice.Lattice`.
	"""


class LatticeViewType(ABC):

	"""
	This abstract base class is a used to represent the LatticeView class in type hints. For a detailed
	description see :class:`blueprints.utils.view.LatticeView`.
	"""


class ColorType(ABC):

	"""
	This abstract base class is a used to represent the View class in type hints. For a detailed
	description see :class:`blueprints.utils.view.View`.
	"""

	_COLORS = {'red':         [1.0,  0.0,   0.0],
		   'orange':      [1.0,  0.5,   0.0],
		   'yellow':      [0.8,  0.8,   0.0],
		   'green':       [0.0,  1.0,   0.0],
		   'teal':        [0.0,  1.0,   1.0],
		   'blue':        [0.0,  0.0,   1.0],
		   'purple':      [1.0,  0.0,   1.0],
		   'black':       [0.0,  0.0,   0.0],
		   'grey':        [0.5,  0.5,   0.5],
		   'white':       [1.0,  1.0,   1.0], 
		   'dark_red':    [0.5,  0.0,   0.0],
		   'dark_orange': [0.5,  0.25,  0.0],
		   'dark_yellow': [0.4,  0.4,   0.0],
		   'dark_green':  [0.0,  0.5,   0.0],
		   'dark_teal':   [0.0,  0.5,   0.5],
		   'dark_blue':   [0.0,  0.0,   0.5],
		   'dark_purple': [0.5,  0.0,   0.5],
		   'dark_grey':   [0.25, 0.25,  0.25]}


class ThingType(ABC):

	"""
	This abstract base class is a used to represent the Thing class in type hints. For a detailed
	description see :class:`blueprints.thing.base.BaseThing`.
	"""

	_NEW_NO_COPY_ATTR   = {'parent'}
	_NEW_DEFAULT_VALS   = dict()
	_NEW_DERIVED_ATTR   = dict()
	_NEW_MUJOCO_ATTR    = {'name':   str}



ThingType._NEW_BLUEPRINT_ATTR = {'parent': ThingType}



class NodeThingType(ThingType):

	"""
	This abstract base class is a used to represent the NodeThing class in type hints. For a detailed
	description see :class:`blueprints.thing.node.NodeThing`.
	"""

	_CYCLE_REF = dict()


class MoveableThingType(ThingType):

	"""
	This abstract base class is a used to represent the MoveableThing class in type hints. For a detailed
	description see :class:`blueprints.thing.moveable.MoveableThing`.
	"""
	_NEW_STEP_CACHE     = {'pos', 
			       'rotation_matrix', 
			       'global_rotation_matrix', 
			       'euler', 
			       'global_pos', 
			       'vel', 
			       'angular_vel'}
	_NEW_DEFAULT_VALS   = {'pos':   np.array([0., 0., 0.], dtype=np.float32),
			       'euler': np.array([0., 0., 0.], dtype=np.float32),
			       'alpha': 0.,
			       'beta':  0.,
			       'gamma': 0.}
	_NEW_BLUEPRINT_ATTR = {'pos':   np.ndarray,
			       'global_pos': np.ndarray, 
			       'alpha': float,
			       'beta':  float,
			       'gamma': float}
	_NEW_MUJOCO_ATTR    = {'euler':    np.ndarray,
			       'pos':      np.ndarray}


class ColoredThingType(ThingType):

	"""
	This abstract base class is a used to represent the ColoredThing class in type hints. For a detailed
	description see :class:`blueprints.thing.colored.ColoredThing`.
	"""

	_NEW_DEFAULT_VALS   = {'rgba':    np.array([.5, .5, .5, 1.], dtype=np.float32),
			       'opacity': 1.}
	_NEW_BLUEPRINT_ATTR = {'color': ColorType}
	_NEW_MUJOCO_ATTR    = {'rgba':  np.ndarray}


class CyclicalThingType(ThingType):

	"""
	This abstract base class is a used to represent the CyclicalThing class in type hints. For a detailed
	description see :class:`blueprints.thing.cyclical.CyclicalThing`.
	"""


class UniqueThingType(ThingType):

	"""
	This abstract base class is a used to represent the UniqueThing class in type hints. For a detailed
	description see :class:`blueprints.thing.unique.UniqueThing`.
	"""


class FocalThingType(ThingType):

	"""
	This abstract base class is a used to represent the FocalThing class in type hints. For a detailed
	description see :class:`blueprints.thing.unique.FocalThing`.
	"""


class CacheType(UniqueThingType):

	"""
	This abstract base class is a used to represent the Cache class in type hints. For a detailed
	description see :class:`blueprints.cache.BaseCache`.
	"""

	_REFERENCE_NAME = 'cache'


class MeshCacheType(CacheType):

	"""
	This abstract base class is a used to represent the MeshCache class in type hints. For a detailed
	description see :class:`blueprints.cache.MeshCache`.
	"""

	_NEW_BLUEPRINT_ATTR = {'vertecies':      list,
			       'faces':          list,
			       'texcoords':      list,
			       'texcoords_idx':  dict,
			       'normals_idx':    dict,
			       'face_normals':   list,
			       'vertex_normals': list,
			       'filename':       str,
			       'centered':       bool}
	_NEW_MUJOCO_ATTR    = {'vertex':   np.ndarray,
			       'face':     np.ndarray}
	_ASSET_OBJ          =  'mesh'


class HFieldCacheType(CacheType):

	"""
	This abstract base class is a used to represent the MeshCache class in type hints. For a detailed
	description see :class:`blueprints.cache.MeshCache`.
	"""

	_NEW_BLUEPRINT_ATTR = {'terrain':   list,
			       'filename':  str}
	_NEW_MUJOCO_ATTR    = {'elevation': np.ndarray, 
			       'nrow':      int, 
			       'ncol':      int}
	_ASSET_OBJ          =  'hfield'


class AssetType(UniqueThingType):

	"""
	This abstract base class is a used to represent the Asset class in type hints. For a detailed
	description see :class:`blueprints.assets.BaseAsset`.
	"""

	_REFERENCE_NAME = 'asset'



class TextureAssetType(AssetType):

	"""
	This abstract base class is a used to represent the TextureAsset class in type hints. For a detailed
	description see :class:`blueprints.assets.TextureAsset`.
	"""

	_NEW_MUJOCO_ATTR    = {'type':         str, 
			       #'colorspace':  str, 
			       'content_type': str, 
			       'file':         str, 
			       'gridsize':     np.ndarray, 
			       'gridlayout':   str, 
			       'fileright':    str, 
			       'fileleft':     str, 
			       'fileup':       str, 
			       'filedown':     str, 
			       'filefront':    str, 
			       'fileback':     str, 
			       'builtin':      str, 
			       'rgb1':         np.ndarray, 
			       'rgb2':         np.ndarray, 
			       'mark':         str, 
			       'markrgb':      np.ndarray, 
			       'random':       float, 
			       'width':        int, 
			       'height':       int, 
			       'hflip':        bool, 
			       'vflip':        bool, 
			       'nchannel':     int}
	_NEW_BLUEPRINT_ATTR = {#'color_space': str, 
			       'content':      str, 
			       'file':         str, 
			       'grid_size':    list, 
			       'grid_layout':  str, 
			       'fileright':    str, 
			       'fileleft':     str, 
			       'fileup':       str, 
			       'filedown':     str, 
			       'filefront':    str, 
			       'fileback':     str, 
			       'builtin':      str, 
			       'color_1':      ColorType, 
			       'color_2':      ColorType, 
			       'mark':         str, 
			       'color_mark':   ColorType, 
			       'random':       float, 
			       'width':        int, 
			       'height':       int, 
			       'h_flip':       bool, 
			       'v_flip':       bool, 
			       'n_channel':    int}
	_NEW_DEFAULT_VALS   = {'gridsize':     np.array([1, 1], dtype=np.int8), 
			       'gridlayout':   '…………',  
			       'rgb1':         np.array([0.8, 0.8, 0.8], dtype=np.float32), 
			       'rgb2':         np.array([0.5, 0.5, 0.5], dtype=np.float32), 
			       'markrgb':      np.array([0.0, 0.0, 0.0], dtype=np.float32), 
			       'random':       0.01, 
			       'width':        0, 
			       'height':       0, 
			       'hflip':        False, 
			       'vflip':        False, 
			       'nchannel':     3}
	_REFERENCE_NAME = 'asset'



class TextureType(ThingType):

	"""
	This abstract base class is a used to represent the Texture class in type hints. For a detailed
	description see :class:`blueprints.texture.BaseTexture`.
	"""
	_NEW_NO_COPY_ATTR      = {'asset'}
	_NEW_SINGLE_CHILD_ATTR = {'asset':       AssetType}
	_NEW_BLUEPRINT_ATTR    = {'asset':       TextureAssetType}



class PlaneTextureType(TextureType):

	"""
	This abstract base class is a used to represent the Texture class in type hints. For a detailed
	description see :class:`blueprints.texture.Plane`.
	"""

	_TYPE = '2d'



class BoxTextureType(TextureType):

	"""
	This abstract base class is a used to represent the Texture class in type hints. For a detailed
	description see :class:`blueprints.texture.Box`.
	"""

	_TYPE = 'cube'



class SkyboxTextureType(BoxTextureType):

	"""
	This abstract base class is a used to represent the Texture class in type hints. For a detailed
	description see :class:`blueprints.texture.Skybox`.
	"""

	_TYPE = 'skybox'



class MaterialAssetType(AssetType, ColoredThingType):

	"""
	This abstract base class is a used to represent the MaterialAsset class in type hints. For a detailed
	description see :class:`blueprints.assets.MaterialAsset`.
	"""

	_NEW_MUJOCO_ATTR    = {'texrepeat':   np.ndarray, 
			       'texuniform':  bool, 
			       'emission':    float, 
			       'specular':    float, 
			       'shininess':   float, 
			       'reflectance': float, 
			       'metallic':    float, 
			       'roughness':   float}
	_NEW_BLUEPRINT_ATTR = {'texture':     TextureType, 
			       'texrepeat':   np.ndarray, 
			       'texuniform':  bool, 
			       'emission':    float, 
			       'specular':    float, 
			       'shininess':   float, 
			       'reflectance': float, 
			       'metallic':    float, 
			       'roughness':   float}
	_NEW_DEFAULT_VALS   = {'texrepeat':   np.array([1., 1.], dtype=np.float32), 
			       'texuniform':  False, 
			       'emission':    0.0, 
			       'specular':    0.5, 
			       'shininess':   0.5, 
			       'reflectance': 0.0, 
			       'metallic':   -1.0, 
			       'roughness':  -1.0}



class MaterialType(NodeThingType):

	"""
	This abstract base class is a used to represent the Material class in type hints. For a detailed
	description see :class:`blueprints.Material`.
	"""

	_NEW_NO_COPY_ATTR      = {'asset'}
	_NEW_SINGLE_CHILD_ATTR = {'asset':       AssetType}
	_NEW_BLUEPRINT_ATTR    = {'texture':     TextureAssetType, 
				  'texrepeat':   np.ndarray, 
				  'texuniform':  bool, 
				  'emission':    float, 
				  'specular':    float, 
				  'shininess':   float, 
				  'reflectance': float, 
				  'metallic':    float, 
				  'roughness':   float}
	



class MeshAssetType(MoveableThingType, AssetType):

	"""
	This abstract base class is a used to represent the MeshAsset class in type hints. For a detailed
	description see :class:`blueprints.assets.MeshAsset`.
	"""

	_NEW_DEFAULT_VALS   = {'scale':    np.array([1., 1., 1.], dtype=np.float32),
			       'refpos':   np.array([0., 0., 0.], dtype=np.float32)}
	_NEW_DERIVED_ATTR   = {'vertex':   np.ndarray}
	_NEW_BLUEPRINT_ATTR = {'scale':    np.ndarray,
			       'filename': str,
			       'cache':    CacheType,
			       'xml_data': bool}
	_NEW_MUJOCO_ATTR    = {'refpos':   np.ndarray,
			       'scale':    np.ndarray}
	_DEL_MUJOCO_ATTR    = {'euler',
			       'pos'}
	_ASSET_OBJ          =  'mesh'


class HFieldAssetType(MoveableThingType, AssetType):

	"""
	This abstract base class is a used to represent the MeshAsset class in type hints. For a detailed
	description see :class:`blueprints.assets.MeshAsset`.
	"""

	_NEW_DEFAULT_VALS   = {'pos':           np.array([0., 0., 0.], dtype=np.float32), 
			       'x_length':      1., 
			       'y_length':      1., 
			       'z_length':      1., 
			       'height_offset': 1.}
	_NEW_DERIVED_ATTR   = {'elevation':     np.ndarray}
	_NEW_MUJOCO_ATTR    = {'size':          np.ndarray}
	_DEL_MUJOCO_ATTR    = {'pos'}
	_NEW_BLUEPRINT_ATTR = {'filename':      str,
			       'cache':         CacheType, 
			       'xml_data':      bool, 
			       'x_length':      float, 
			       'y_length':      float, 
			       'z_length':      float}
	_ASSET_OBJ          =  'hfield'


class WorldType(NodeThingType):

	"""
	This abstract base class is a used to represent the World class in type hints. For a detailed
	description see :class:`blueprints.world.World`.
	"""

	_MUJOCO_OBJ = 'world'


class BodyType(NodeThingType):

	"""
	This abstract base class is a used to represent the Body class in type hints. For a detailed
	description see :class:`blueprints.body.Body`.
	"""

	_NEW_MUJOCO_ATTR = {'euler': np.ndarray,
		            'pos':   np.ndarray}
	_MUJOCO_OBJ      =  'body'
	_MUJOCO_DATA     =  'body'


class PlaceholderType(MoveableThingType):

	"""
	This abstract base class is a used to represent the Placeholder class in type hints. For a detailed
	description see :class:`blueprints.placeholder.Placeholder`.
	"""


class TubeType(ThingType):

	"""
	This abstract base class is a used to represent the Tube class in type hints. For a detailed
	description see :class:`blueprints.tube.BaseTube`.
	"""


class GeomType(MoveableThingType, ColoredThingType, NodeThingType):

	"""
	This abstract base class is a used to represent the Geom class in type hints. For a detailed
	description see :class:`blueprints.geoms.BaseGeom`.
	"""

	_NEW_SINGLE_CHILD_ATTR = {'material':           MaterialType}
	_NEW_DEFAULT_VALS      = {'friction':           np.array([1.0, 0.005, 0.0001], dtype=np.float32),
				  'margin':             0.,
				  'gap':                0.,
				  'shellinertia':       False,
				  'mass':               None,
				  'density':            1000.}
	_NEW_DERIVED_ATTR      = {'type':               str}
	_NEW_BLUEPRINT_ATTR    = {'mass':               float,
				  'density':            float,
				  'shellinertia':       bool,
				  'margin':             float,
				  'gap':                float,
				  'sliding_friction':   float,
				  'torsional_friction': float,
				  'rolling_friction':   float, 
				  'material':           MaterialType}
	_NEW_MUJOCO_ATTR       = {'size':               np.ndarray,
				  'mass':               float,
				  'density':            float,
				  'shellinertia':       bool,
				  'margin':             float,
				  'gap':                float,
				  'friction':           np.ndarray,
				  'type':               str}
	_MUJOCO_OBJ             = 'geom'
	_MUJOCO_DATA            = 'geom'


class CapsuleGeomType(GeomType, TubeType):

	"""
	This abstract base class is a used to represent the CapsuleGeom class in type hints. For a detailed
	description see :class:`blueprints.geoms.Capsule`.
	"""

	_NEW_BLUEPRINT_ATTR = {'radius': float,
			       'length': float}


class CylinderGeomType(GeomType, TubeType):

	"""
	This abstract base class is a used to represent the CylinderGeom class in type hints. For a detailed
	description see :class:`blueprints.geoms.Cylinder`.
	"""

	_NEW_BLUEPRINT_ATTR = {'radius': float,
			       'length': float}


class BoxGeomType(GeomType, TubeType):

	"""
	This abstract base class is a used to represent the BoxGeom class in type hints. For a detailed
	description see :class:`blueprints.geoms.Box`.
	"""

	_NEW_BLUEPRINT_ATTR = {'x_length': float,
			       'y_length': float,
			       'z_length': float}


class PlaneGeomType(GeomType):

	"""
	This abstract base class is a used to represent the PlaneGeom class in type hints. For a detailed
	description see :class:`blueprints.geoms.Plane`.
	"""

	_NEW_BLUEPRINT_ATTR = {'x_length': float,
			       'y_length': float,
			       'spacing':  float}


class SphereGeomType(GeomType):

	"""
	This abstract base class is a used to represent the SphereGeom class in type hints. For a detailed
	description see :class:`blueprints.geoms.Sphere`.
	"""

	_NEW_BLUEPRINT_ATTR = {'radius': float}


class EllipsoidGeomType(GeomType):

	"""
	This abstract base class is a used to represent the EllipsoidGeom class in type hints. For a detailed
	description see :class:`blueprints.geoms.Ellipsoid`.
	"""

	_NEW_BLUEPRINT_ATTR = {'x_radius': float,
			       'y_radius': float,
			       'z_radius': float}


class MeshGeomType(GeomType):

	"""
	This abstract base class is a used to represent the MeshGeom class in type hints. For a detailed
	description see :class:`blueprints.geoms.Mesh`.
	"""

	_NEW_SINGLE_CHILD_ATTR = {'asset':    AssetType, 
				  'material': MaterialType}
	_NEW_NO_COPY_ATTR      = {'asset'}
	_NEW_BLUEPRINT_ATTR    = {'asset': AssetType}
	#_DEL_BLUEPRINT_ATTR = {'pos'}
	_DEL_MUJOCO_ATTR       = {'pos',
				  'size'}
	_MUJOCO_DATA           =  'mesh'


class HFieldGeomType(GeomType):

	"""
	This abstract base class is a used to represent the HfieldGeom class in type hints. For a detailed
	description see :class:`blueprints.geoms.Hfield`.
	"""

	_NEW_SINGLE_CHILD_ATTR = {'asset': AssetType}
	_NEW_BLUEPRINT_ATTR    = {'asset': AssetType}
	_DEL_MUJOCO_ATTR       = {'size'}


class SiteType(MoveableThingType, NodeThingType):

	"""
	This abstract base class is a used to represent the Site class in type hints. For a detailed
	description see :class:`blueprints.sites.BaseSite`.
	"""

	_NEW_DEFAULT_VALS   = {'size': np.array([0.005, 0.005, 0.005], dtype=np.float32)}
	_NEW_DERIVED_ATTR   = {'type': str}
	_NEW_MUJOCO_ATTR    = {'size': np.ndarray,
			       'rgba': np.ndarray,
			       'type': str}
	_MUJOCO_OBJ	    =  'site'
	_MUJOCO_DATA        =  'site'


class CapsuleSiteType(TubeType, SiteType):

	"""
	This abstract base class is a used to represent the CapsuleSite class in type hints. For a detailed
	description see :class:`blueprints.sites.Capsule`.
	"""

	_NEW_BLUEPRINT_ATTR = {'radius': float,
			       'length': float}


class CylinderSiteType(TubeType, SiteType):

	"""
	This abstract base class is a used to represent the CylinderSite class in type hints. For a detailed
	description see :class:`blueprints.sites.Cylinder`.
	"""

	_NEW_BLUEPRINT_ATTR = {'radius': float,
			       'length': float}


class BoxSiteType(TubeType, SiteType):

	"""
	This abstract base class is a used to represent the BoxSite class in type hints. For a detailed
	description see :class:`blueprints.sites.Box`.
	"""

	_NEW_BLUEPRINT_ATTR = {'x_length': float,
			       'y_length': float,
			       'z_length': float}


class SphereSiteType(SiteType):

	"""
	This abstract base class is a used to represent the SphereSite class in type hints. For a detailed
	description see :class:`blueprints.sites.Sphere`.
	"""

	_NEW_BLUEPRINT_ATTR = {'radius': float}


class EllipsoidSiteType(SiteType):

	"""
	This abstract base class is a used to represent the EllipsoidSite class in type hints. For a detailed
	description see :class:`blueprints.sites.Ellipsoid`.
	"""

	_NEW_BLUEPRINT_ATTR = {'x_radius': float,
			       'y_radius': float,
			       'z_radius': float}


class JointType(NodeThingType):

	"""
	This abstract base class is a used to represent the Joint class in type hints. For a detailed
	description see :class:`blueprints.joints.BaseJoint`.
	"""

	_NEW_DEFAULT_VALS   = {'springdamper':       np.array([0., 0.], dtype=np.float32),
			       'actuatorforcerange': np.array([0., 0.], dtype=np.float32),
			       'stiffness':          0.,
			       'springref':          0.,
			       'armature':           0.,
			       'damping':            0.}
	_NEW_DERIVED_ATTR   = {'type':               str}
	_NEW_BLUEPRINT_ATTR = {'springdamper':       np.ndarray,
			       'actuatorforcerange': np.ndarray,
			       'stiffness':          np.ndarray,
			       'springref':          np.ndarray,
			       'armature':           np.ndarray,
			       'damping':            np.ndarray,
			       #'sensors':      list,
			       #'actuators':    list,
			       }
	_NEW_MUJOCO_ATTR    = {'springdamper':       np.ndarray,
			       'actuatorforcerange': np.ndarray,
			       'stiffness':          np.ndarray,
			       'springref':          np.ndarray,
			       'armature':           np.ndarray,
			       'damping':            np.ndarray,
			       'type':               str}
	_DEL_MUJOCO_ATTR    = {'euler'}
	_DEL_BLUEPRINT_ATTR = {'euler'}
	_MUJOCO_OBJ         =  'joint'
	_MUJOCO_DATA        =  'jnt'


class HingeType(JointType):

	"""
	This abstract base class is a used to represent the Hinge class in type hints. For a detailed
	description see :class:`blueprints.joints.Hinge`.
	"""

	_NEW_DEFAULT_VALS   = {'axis':        [0., 0., 1.],
			       'range':       [0., 0.],
			       'ref':          0,
			       'frictionloss': 0.}
	_NEW_BLUEPRINT_ATTR = {'axis':         np.ndarray,
			       'range':        np.ndarray,
			       'ref':          float,
			       'frictionloss': float}
	_NEW_MUJOCO_ATTR    = {'axis':         np.ndarray,
			       'range':        np.ndarray,
			       'ref':          float,
			       'frictionloss': float}


class SlideType(JointType):

	"""
	This abstract base class is a used to represent the Slide class in type hints. For a detailed
	description see :class:`blueprints.joints.Slide`.
	"""

	_NEW_DEFAULT_VALS   = {'axis':        [0., 0., 1.],
			       'range':       [0., 0.],
			       'ref':          0,
			       'frictionloss': 0.}
	_NEW_BLUEPRINT_ATTR = {'axis':         np.ndarray,
			       'range':        np.ndarray,
			       'ref':          float,
			       'frictionloss': float}
	_NEW_MUJOCO_ATTR    = {'axis':         np.ndarray,
			       'range':        np.ndarray,
			       'ref':          float,
			       'frictionloss': float}


class BallType(JointType):

	"""
	This abstract base class is a used to represent the Ball class in type hints. For a detailed
	description see :class:`blueprints.joints.Ball`.
	"""

	_NEW_DEFAULT_VALS   = {'range':       [0., 0.],
			       'frictionloss': 0.}
	_NEW_BLUEPRINT_ATTR = {'range':        np.ndarray,
			       'frictionloss': float}
	_NEW_MUJOCO_ATTR    = {'range':        np.ndarray,
			       'frictionloss': float}


class FreeType(JointType):

	"""
	This abstract base class is a used to represent the Free class in type hints. For a detailed
	description see :class:`blueprints.joints.Free`.
	"""


class SensorType(ThingType):

	"""
	This abstract base class is a used to represent the Sensor class in type hints. For a detailed
	description see :class:`blueprints.sensors.BaseSensor`.
	"""

	_NEW_DEFAULT_VALS   = {'noise':  0.,
			       'cutoff': 0.}
	_NEW_BLUEPRINT_ATTR = {'noise':  float,
			       'cutoff': float}
	_NEW_MUJOCO_ATTR    = {'noise':  float,
			       'cutoff': float}
	_REFERENCE_TYPES    = {ThingType}
	_MUJOCO_OBJ	    =  'sensor'


class SiteSensorType(SensorType):

	"""
	This abstract base class is a used to represent the SiteSensor class in type hints. For a detailed
	description see :class:`blueprints.sensors.SiteSensor`.
	"""

	_PARENT_TYPE     =  'site'
	_DERIVED_ATTR    = {'site': str}
	_NEW_MUJOCO_ATTR = {'site': str}


class JointSensorType(SensorType):

	"""
	This abstract base class is a used to represent the JointSensor class in type hints. For a detailed
	description see :class:`blueprints.sensors.JointSensor`.
	"""

	_PARENT_TYPE     =  'joint'
	_DERIVED_ATTR    = {'joint': str}
	_NEW_MUJOCO_ATTR = {'joint': str}


class ActuatorSensorType(SensorType):

	"""
	This abstract base class is a used to represent the ActuatorSensor class in type hints. For a detailed
	description see :class:`blueprints.sensors.ActuatorSensor`.
	"""

	_PARENT_TYPE     =  'actuator'
	_DERIVED_ATTR    = {'actuator': str}
	_NEW_MUJOCO_ATTR = {'actuator': str}


class TendonType(FocalThingType):

	"""
	This abstract base class is a used to represent the Tendon class in type hints. For a detailed
	description see :class:`blueprints.tendon.Tendon`.
	"""

	_FIXED_ATTR         = {'name', 
			       'limited', 
			       'range', 
			       'solreflimit', 
			       'solimplimit', 
			       'solreffriction', 
			       'solimpfriction', 
			       'frictionloss', 
			       'margin', 
			       'springlength', 
			       'stiffness', 
			       'damping'}
	_NEW_MUJOCO_ATTR    = {'limited':            bool, 
			       'actuatorfrclimited': bool, 
			       'range':              np.ndarray, 
			       'actuatorfrcrange':   np.ndarray, 
			       'frictionloss':       float, 
			       'width':              float, 
			       'stiffness':          float, 
			       'damping':            float, 
			       'armature':           float}
	_NEW_DEFAULT_VALS   = {'limited':            None, 
			       'actuatorfrclimited': None, 
			       'range':              np.array([0., 0.], dtype=np.float32), 
			       'actuatorfrcrange':   np.array([0., 0.], dtype=np.float32), 
			       'frictionloss':       0., 
			       'width':              0.003, 
			       'stiffness':          0., 
			       'damping':            0., 
			       'armature':           0.}
	_NEW_BLUEPRINT_ATTR = {'limited':            bool,
			       'act_force_limited':  bool, 
			       'min_length':         float,
			       'max_length':         float, 
			       'min_act_force':      float,
			       'max_act_force':      float, 
			       'frictionloss':       float, 
			       'width':              float, 
			       'stiffness':          float, 
			       'damping':            float, 
			       'armature':           float,}



class InfoLaserType(SiteSensorType):

	"""
	This abstract base class is a used to represent the InfoLaser class in type hints. For a detailed
	description see :class:`blueprints.sensors.InfoLaser`.
	"""

	_DIMENSION          = -1
	_NEW_BLUEPRINT_ATTR = {'axis': str}
	_NEW_DEFAULT_VALS   = {'axis': np.array([0., 0., 1.], dtype=np.float32)}


class ActuatorType(CyclicalThingType, NodeThingType):

	"""
	This abstract base class is a used to represent the Actuator class in type hints. For a detailed
	description see :class:`blueprints.actuators.BaseActuator`.
	"""

	_NEW_DEFAULT_VALS   = {'ctrllimited':  None,
			       'forcelimited': None,
			       'actlimited':   None,
			       'ctrlrange':    np.array([0., 0.], dtype=np.float32),
			       'forcerange':   np.array([0., 0.], dtype=np.float32),
			       'actrange':     np.array([0., 0.], dtype=np.float32),
			       'lengthrange':  np.array([0., 0.], dtype=np.float32),
			       'gear':         np.array([1., 0., 0., 0., 0., 0.], dtype=np.float32),
			       'cranklength':  0.,
			       'dyntype':      'none',
			       'gaintype':     'fixed',
			       'biastype':     'none',
			       'dynprm':       np.array(1., dtype=np.float32),
			       'gainprm':      np.array(1., dtype=np.float32),
			       'biasprm':      np.array(1., dtype=np.float32)}
	_NEW_BLUEPRINT_ATTR = {'ctrllimited':  bool,
			       'forcelimited': bool,
			       'actlimited':   bool,
			       'ctrlrange':    np.ndarray,
			       'forcerange':   np.ndarray,
			       'actrange':     np.ndarray,
			       'lengthrange':  np.ndarray,
			       'gear':         np.ndarray,
			       'cranklength':  float,
			       'dyntype':      str,
		     	       'gaintype':     str,
		     	       'biastype':     str,
		     	       'dynprm':       np.ndarray,
		     	       'gainprm':      np.ndarray,
		     	       'biasprm':      np.ndarray,
		     	       'inparent':     bool}
	_NEW_MUJOCO_ATTR    = {'ctrllimited':  bool,
			       'forcelimited': bool,
			       'actlimited':   bool,
			       'ctrlrange':    np.ndarray,
			       'forcerange':   np.ndarray,
			       'actrange':     np.ndarray,
			       'lengthrange':  np.ndarray,
			       'gear':         np.ndarray,
			       'cranklength':  float,
			       'dyntype':      str,
		     	       'gaintype':     str,
		     	       'biastype':     str,
		     	       'dynprm':       np.ndarray,
		     	       'gainprm':      np.ndarray,
		     	       'biasprm':      np.ndarray}
	_OTHER_REFERENCES   = {'refsite':      'ref_actuators'}
	_PARENT_REFERENCE   =  'refsite'
	_MUJOCO_OBJ         =  'actuator'


class PositionType(ActuatorType):

	"""
	This abstract base class is a used to represent the Actuator class in type hints. For a detailed
	description see :class:`blueprints.actuators.Position`.
	"""

	_NEW_DEFAULT_VALS   = {'kp': 1.}
	_NEW_BLUEPRINT_ATTR = {'kp': float}
	_NEW_MUJOCO_ATTR    = {'kp': float}


class VelocityType(ActuatorType):

	"""
	This abstract base class is a used to represent the Actuator class in type hints. For a detailed
	description see :class:`blueprints.actuators.Velocity`.
	"""

	_NEW_DEFAULT_VALS   = {'kv': 1.}
	_NEW_BLUEPRINT_ATTR = {'kv': float}
	_NEW_MUJOCO_ATTR    = {'kv': float}


class IntVelocityType(ActuatorType):

	"""
	This abstract base class is a used to represent the Actuator class in type hints. For a detailed
	description see :class:`blueprints.actuators.IntVelocity`.
	"""

	_NEW_DEFAULT_VALS   = {'kp': 1.}
	_NEW_BLUEPRINT_ATTR = {'kp': float}
	_NEW_MUJOCO_ATTR    = {'kp': float}


class DamperType(ActuatorType):

	"""
	This abstract base class is a used to represent the Actuator class in type hints. For a detailed
	description see :class:`blueprints.actuators.Damper`.
	"""

	_NEW_DEFAULT_VALS   = {'kv': 1.}
	_NEW_BLUEPRINT_ATTR = {'kv': float}
	_NEW_MUJOCO_ATTR    = {'kv': float}


class CylinderType(ActuatorType):

	"""
	This abstract base class is a used to represent the Actuator class in type hints. For a detailed
	description see :class:`blueprints.actuators.Cylinder`.
	"""

	_NEW_DEFAULT_VALS   = {'timeconst': 1.,
			       'area':      1.,
			       'diameter':  None,
			       'bias': np.zeros(3, dtype=np.float32)}
	_NEW_BLUEPRINT_ATTR = {'timeconst': float,
			       'area':      float,
			       'diameter':  float|None,
			       'bias':      np.ndarray}
	_NEW_MUJOCO_ATTR    = {'timeconst': float,
			       'area':      float,
			       'diameter':  float|None,
			       'bias':      np.ndarray}


class MuscleType(ActuatorType):

	"""
	This abstract base class is a used to represent the Actuator class in type hints. For a detailed
	description see :class:`blueprints.actuators.Muscle`.
	"""

	_NEW_DEFAULT_VALS   = {'timeconst': np.array([0.01, 0.04], dtype=np.float32),
			       'tausmooth': 0.,
			       'range':     np.array([0.75, 1.05], dtype=np.float32),
			       'force':    -1.,
			       'scale':     200,
			       'lmin':      0.5,
			       'lmax':      1.6,
			       'vmax':      1.5,
			       'fpmax':     1.3,
			       'fvmax':     1.2}
	_NEW_BLUEPRINT_ATTR = {'timeconst': np.ndarray,
			       'tausmooth': float,
			       'range':     np.ndarray,
			       'force':     float,
			       'scale':     float,
			       'lmin':      float,
			       'lmax':      float,
			       'vmax':      float,
			       'fpmax':     float,
			       'fvmax':     float}
	_NEW_MUJOCO_ATTR    = {'timeconst': np.ndarray,
			       'tausmooth': float,
			       'range':     np.ndarray,
			       'force':     float,
			       'scale':     float,
			       'lmin':      float,
			       'lmax':      float,
			       'vmax':      float,
			       'fpmax':     float,
			       'fvmax':     float}


class AdhesionType(ActuatorType):

	"""
	This abstract base class is a used to represent the Actuator class in type hints. For a detailed
	description see :class:`blueprints.actuators.Adhesion`.
	"""

	_NEW_DEFAULT_VALS   = {'gain': 1.}
	_NEW_BLUEPRINT_ATTR = {'gain': float}
	_NEW_MUJOCO_ATTR    = {'gain': float}


class LightType(CyclicalThingType, MoveableThingType):

	"""
	This abstract base class is a used to represent the Light class in type hints. For a detailed
	description see :class:`blueprints.light.Light`.
	"""

	_NEW_DEFAULT_VALS   = {'pos':         np.array([ 0., 0., 0.], dtype=np.float32),
			       'dir':         np.array([ 0., 0.,-1.], dtype=np.float32),
			       'attenuation': np.array([ 1., 0., 0.], dtype=np.float32),
			       'ambient':     np.array([ 0., 0., 0.], dtype=np.float32),
			       'diffuse':     np.array([ .7, .7, .7], dtype=np.float32),
			       'specular':    np.array([ .3, .3, .3], dtype=np.float32),
			       'mode':        'fixed',
			       'directional': False,
			       'castshadow':  True,
			       'active':      True,
			       'cutoff':      3.1415926535897932384626433832/2,
			       'exponent':    10.}
	_NEW_BLUEPRINT_ATTR = {'dir':         np.ndarray,
			       'attenuation': np.ndarray,
			       'ambient':     np.ndarray,
			       'diffuse':     np.ndarray,
			       'specular':    np.ndarray,
			       'mode':        str,
			       'directional': bool,
			       'castshadow':  bool,
			       'active':      bool,
			       'cutoff':      float,
			       'exponent':    float}
	_NEW_MUJOCO_ATTR    = {'dir':         np.ndarray,
			       'attenuation': np.ndarray,
			       'ambient':     np.ndarray,
			       'diffuse':     np.ndarray,
			       'specular':    np.ndarray,
			       'mode':        str,
			       'directional': bool,
			       'castshadow':  bool,
			       'active':      bool,
			       'cutoff':      float,
			       'exponent':    float}
	_DEL_MUJOCO_ATTR    = {'euler'}
	_MODES              = {'fixed',
			       'track',
			       'trackcom',
			       'targetbody',
			       'targetbodycom'}
	_OTHER_REFERENCES   = {'target': 'targeting_lights'}
	_PARENT_REFERENCE   =  'target'
	_MUJOCO_OBJ         =  'light'
	_MUJOCO_DATA        =  'light'


class CameraType(CyclicalThingType):

	"""
	This abstract base class is a used to represent the Camera class in type hints. For a detailed
	description see :class:`blueprints.camera.Camera`.
	"""

	_NEW_DEFAULT_VALS   = {'mode':        'fixed',
			       'resolution':  [1, 1], 
			       'fovy':        45.0,
			       'ipd':         0.068}
	_NEW_BLUEPRINT_ATTR = {'mode':        str,
			       'resolution':  np.ndarray, 
			       'fieldofview': float,
			       'ipd':         float, 
			       'fps':         int}
	_NEW_MUJOCO_ATTR    = {'mode':        str,
			       'resolution':  np.ndarray, 
			       'fovy':        float,
			       'ipd':         float}
	_MODES              = {'fixed',
			       'track',
			       'trackcom',
			       'targetbody',
			       'targetbodycom'}
	_OTHER_REFERENCES   = {'target': 'targeting_cameras'}
	_PARENT_REFERENCE   =  'target'
	_MUJOCO_OBJ         =  'camera'


class AgentType(BodyType):

	"""
	This abstract base class is a used to represent the Agent class in type hints. For a detailed
	description see :class:`blueprints.agent.Agent`.
	"""
