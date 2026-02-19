# Blueprints
Blueprints is a pythonic library that constructs Mujoco simulations and gives access to runtime data. It is designed to simplify access to mujoco and provide helpful subroutines for procedural generation, aiding environment randomization, curriculum learning, hindsight experiance replay and other RL paradigms aiming to manipulate the environment.

Once a certain structure of Mujoco elements has been build, it can be copied, many times throughout the world. The same objects through which the simulation has been constructed also serve as access objects to the simulations runtime data, no need of keeping track of indecies for ``mj_data.obj_type(index)`` calls.

Blueprints offers many conveniences for maipulating many objects attributes at once, implements standart agent interfaces for RL and bundles all sorts of data access in a finegrained indexing scheme through Views.

## Installation
Directly from PyPI:
```
$ pip install mujoco_blueprints
```

Alternatively from Github:
```
$ git clone https://github.com/mortimervonchappuis/mujoco_blueprints.git
```

See [documentation](https://mujoco-blueprints.readthedocs.io/en/latest/index.html) for more details.