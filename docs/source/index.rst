.. blueprints documentation master file, created by
   sphinx-quickstart on Mon Jul  1 20:17:36 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

  .. inheritance-diagram:: blueprints.types

Blueprints
==========

Blueprints is a pythonic library that constructs Mujoco simulations and gives access to runtime data. It is designed to simplify access to mujoco and provide helpful subroutines for procedural generation, aiding environment randomization, curriculum learning, hindsight experiance replay and other RL paradigms aiming to manipulate the environment.

Once a certain structure of Mujoco elements has been build, it can be copied, many times throughout the world. The same objects through which the simulation has been constructed also serve as access objects to the simulations runtime data, no need of keeping track of indecies for ``mj_data.obj_type(index)`` calls.

Blueprints offers many conveniences for maipulating many objects attributes at once, implements standart agent interfaces for RL and bundles all sorts of data access in a finegrained indexing scheme through Views.

Installation
------------
Directly from PyPI:

.. code-block::
   :caption: pip

   $ pip install mujoco_blueprints

Alternatively from Github:

.. code-block::
   :caption: git

   $ git clone https://github.com/mortimervonchappuis/mujoco_blueprints.git


.. include:: introduction.rst


Contents
--------

.. toctree::
   :maxdepth: 1

   tutorial
   Blueprints <blueprints>
   Things <blueprints.thing>
   Utils <blueprints.utils>
   introduction

..
   overview
   Texture <blueprints.texture>
   Body <body.rst>


..
   View <blueprints.utils.view>
   Lattice <blueprints.utils.lattice>
   World <blueprints.world>
   Body <blueprints.body>
   Geoms <blueprints.geoms>
   Sites <blueprints.sites>
   Joints <blueprints.joints>
   Actuators <blueprints.actuators>
   Camera <blueprints.camera>
   Sensors <blueprints.sensors>
   blueprints


.. automodapi:: blueprints
	:skip: types.py restrict
	:skip: AssetType
	introduction

.. 
 inheritance-diagram:: blueprints.thing.base.BaseThing
..    :top-classes: blueprints.thing.base.BaseThing

 .. inheritance-diagram:: blueprints.types.ThingType
    :include-subclasses:

..   :top-classes: blueprints.thing.base.BaseThing
..   blueprints.thing.colored.ColoredThing
..    :include-subclasses:

..   :include-subclasses:
 	:include: World

..
 .. automodapi:: blueprints
   :include: Body
   :skip: chain
   :skip: defaultdict
   :no-inheritance-diagram:
   :no-inherited-members:
   :no-heading:
   :no-main-docstr:
.. 	
 blueprints.thing, blueprints.World
 	:include: introduction
   :no-inheritance-diagram:


..
 Indices and tables
 ==================
 
 * :ref:`modindex`
 * :ref:`search`
