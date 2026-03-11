"""
Gymnasium integration for mujoco_blueprints.

Requires gymnasium as an optional dependency::

	pip install gymnasium

or::

	pip install mujoco-blueprints[rl]
"""

try:
	import gymnasium
except ImportError:
	raise ImportError(
		"blueprints.rl requires gymnasium. "
		"Install with: pip install gymnasium\n"
		"  or: pip install mujoco-blueprints[rl]"
	)

from .env import Env

__all__ = ["Env"]
