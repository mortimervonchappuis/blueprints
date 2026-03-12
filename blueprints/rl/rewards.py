"""
Composable reward functions for mujoco_blueprints RL environments.

Each function takes an agent (and optionally an action) and returns a float.
Combine them in your :meth:`Env.reward` method::

	def reward(self, action):
		return (velocity(self.agent, axis='x')
			+ alive_bonus(self.agent, min_z=0.2)
			- action_penalty(action, scale=0.01))
"""

import numpy as np


def velocity(agent, axis='x', scale=1.0):
	"""
	Reward proportional to the agent's velocity along an axis.

	Parameters
	----------
	agent : blueprints.Agent
		The agent whose velocity is measured.
	axis : str
		One of ``'x'``, ``'y'``, ``'z'``. Default: ``'x'``.
	scale : float
		Multiplier applied to the velocity. Default: 1.0.

	Returns
	-------
	float
	"""
	vel = getattr(agent, f'{axis}_vel')
	return scale * float(vel)


def alive_bonus(agent, bonus=1.0, min_z=None, max_z=None):
	"""
	Constant bonus awarded while the agent is alive (within height bounds).

	Parameters
	----------
	agent : blueprints.Agent
		The agent to check.
	bonus : float
		Reward given when alive. Default: 1.0.
	min_z : float or None
		If set, agent must be above this height to be considered alive.
	max_z : float or None
		If set, agent must be below this height to be considered alive.

	Returns
	-------
	float
		``bonus`` if alive, ``0.0`` otherwise.
	"""
	z = float(agent.z)
	if min_z is not None and z < min_z:
		return 0.0
	if max_z is not None and z > max_z:
		return 0.0
	return bonus


def action_penalty(action, scale=0.01, norm='l2'):
	"""
	Penalty proportional to the magnitude of the action.

	Parameters
	----------
	action : np.ndarray
		The action vector.
	scale : float
		Multiplier for the penalty. Default: 0.01.
	norm : str
		``'l2'`` for squared sum, ``'l1'`` for absolute sum. Default: ``'l2'``.

	Returns
	-------
	float
		A non-negative penalty value (to be subtracted from reward).
	"""
	action = np.asarray(action)
	if norm == 'l2':
		return scale * float(np.sum(action ** 2))
	elif norm == 'l1':
		return scale * float(np.sum(np.abs(action)))
	else:
		raise ValueError(f"Unknown norm '{norm}'. Use 'l1' or 'l2'.")


def height_penalty(agent, target_z, scale=1.0):
	"""
	Penalty for deviating from a target height.

	Parameters
	----------
	agent : blueprints.Agent
		The agent to check.
	target_z : float
		Desired height.
	scale : float
		Multiplier. Default: 1.0.

	Returns
	-------
	float
		A non-negative penalty value.
	"""
	return scale * float((agent.z - target_z) ** 2)


def goal_distance(agent, goal, scale=1.0, axes='xy'):
	"""
	Negative distance to a goal position (closer = higher reward).

	Parameters
	----------
	agent : blueprints.Agent
		The agent.
	goal : array_like
		Target position. Length must match ``axes``.
	scale : float
		Multiplier. Default: 1.0.
	axes : str
		Which axes to measure. E.g. ``'xy'``, ``'xyz'``, ``'x'``. Default: ``'xy'``.

	Returns
	-------
	float
		Negative scaled distance (always <= 0).
	"""
	pos = np.array([float(getattr(agent, ax)) for ax in axes])
	goal = np.asarray(goal, dtype=np.float64)
	dist = float(np.linalg.norm(pos - goal))
	return -scale * dist


def contact_cost(agent, scale=0.001):
	"""
	Penalty based on total contact force on the agent's touch sensors.

	Parameters
	----------
	agent : blueprints.Agent
		The agent. Must have Touch sensors attached.
	scale : float
		Multiplier. Default: 0.001.

	Returns
	-------
	float
		A non-negative penalty value.
	"""
	total = 0.0
	for sensor in agent.sensors:
		if sensor.__class__.__name__ == 'Touch':
			total += float(np.sum(sensor.observation ** 2))
	return scale * total
