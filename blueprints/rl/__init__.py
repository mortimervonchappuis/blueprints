"""
RL integration for mujoco_blueprints.

Requires gymnasium as an optional dependency::

	pip install gymnasium

For multi-agent support, also install pettingzoo::

	pip install pettingzoo
"""

try:
	import gymnasium
except ImportError:
	raise ImportError(
		"blueprints.rl requires gymnasium. "
		"Install with: pip install gymnasium\n"
		"  or: pip install mujoco-blueprints[rl]"
	)

from .env      import Env
from .         import rewards
from .         import wrappers

# MultiAgentEnv requires pettingzoo — import lazily
def __getattr__(name):
	if name == "MultiAgentEnv":
		from .multi_env import MultiAgentEnv
		return MultiAgentEnv
	raise AttributeError(f"module 'blueprints.rl' has no attribute {name!r}")

__all__ = ["Env", "MultiAgentEnv", "rewards", "wrappers"]
