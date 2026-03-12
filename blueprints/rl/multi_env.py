"""
PettingZoo ParallelEnv wrapper for mujoco_blueprints multi-agent environments.

Requires pettingzoo as an optional dependency::

	pip install pettingzoo
"""

import numpy as np

try:
	from pettingzoo import ParallelEnv
except ImportError:
	raise ImportError(
		"blueprints.rl.multi_env requires pettingzoo. "
		"Install with: pip install pettingzoo\n"
		"  or: pip install mujoco-blueprints[multi-rl]"
	)

import gymnasium
import blueprints as blue


class MultiAgentEnv(ParallelEnv):
	"""
	Base class for creating PettingZoo parallel environments with mujoco_blueprints.

	All agents share one :class:`World <blueprints.world.World>` and step simultaneously.

	Subclass and implement:

	- :meth:`setup` *(required)* — create ``self.world`` with multiple :class:`Agents <blueprints.agent.Agent>`
	- :meth:`reward` *(required)* — compute per-agent rewards
	- :meth:`terminated` *(optional)* — per-agent termination conditions
	- :meth:`truncated` *(optional)* — per-agent truncation conditions
	- :meth:`info` *(optional)* — per-agent info dicts
	- :meth:`on_reset` *(optional)* — called each reset before ``world.reset()``

	Example::

		from blueprints.rl import MultiAgentEnv
		import blueprints as blue

		class Arena(MultiAgentEnv):
			n_substeps = 4

			def setup(self):
				self.world = blue.World()
				for i in range(3):
					agent = blue.Agent(name=f'robot_{i}', pos=[i, 0, 1])
					joint = blue.joints.Hinge(axis=[0, 0, 1])
					joint.attach(blue.actuators.Motor(gear=[100]),
						     blue.sensors.JointPos())
					agent.attach(blue.geoms.Sphere(size=0.1), joint)
					self.world.attach(agent)

			def reward(self, agent_id, action):
				agent = self._agents_by_id[agent_id]
				return float(agent.x_vel)

	Attributes
	----------
	n_substeps : int
		Number of MuJoCo simulation steps per :meth:`step` call. Default: 1.
	"""

	metadata = {"name": "MultiAgentEnv", "is_parallelizable": True}
	n_substeps = 1

	def __init__(self, render_mode=None, **kwargs):
		self.render_mode = render_mode
		self.setup()
		if not hasattr(self, 'world') or not isinstance(self.world, blue.World):
			raise AttributeError(
				"setup() must set self.world to a blueprints.World instance."
			)
		if not self.world._built:
			self.world.build()
		agent_list = list(self.world.agents)
		if len(agent_list) < 2:
			raise ValueError(
				f"MultiAgentEnv requires at least 2 agents, "
				f"found {len(agent_list)}. Use blueprints.rl.Env for single-agent."
			)
		self.possible_agents = [a.name for a in agent_list]
		self.agents = list(self.possible_agents)
		self._agents_by_id = {a.name: a for a in agent_list}
		self._observation_spaces = {
			a.name: self._build_obs_space(a) for a in agent_list
		}
		self._action_spaces = {
			a.name: self._build_act_space(a) for a in agent_list
		}

	# ── User-implemented methods ──

	def setup(self):
		"""
		Create ``self.world`` with two or more :class:`Agents <blueprints.agent.Agent>`.
		The base class calls ``world.build()`` after this method returns.

		Raises
		------
		NotImplementedError
			If not overridden by the subclass.
		"""
		raise NotImplementedError(
			"Subclasses must implement setup() and set self.world."
		)

	def reward(self, agent_id, action):
		"""
		Compute reward for a single agent.

		Parameters
		----------
		agent_id : str
			The agent's name/id.
		action : np.ndarray
			The action that agent took.

		Returns
		-------
		float

		Raises
		------
		NotImplementedError
			If not overridden by the subclass.
		"""
		raise NotImplementedError(
			"Subclasses must implement reward(agent_id, action)."
		)

	def terminated(self, agent_id):
		"""
		Whether the given agent's episode has terminated.

		Parameters
		----------
		agent_id : str
			The agent's name/id.

		Returns
		-------
		bool
		"""
		return False

	def truncated(self, agent_id):
		"""
		Whether the given agent's episode has been truncated.

		Parameters
		----------
		agent_id : str
			The agent's name/id.

		Returns
		-------
		bool
		"""
		return False

	def info(self, agent_id):
		"""
		Extra info dict for the given agent.

		Parameters
		----------
		agent_id : str
			The agent's name/id.

		Returns
		-------
		dict
		"""
		return {}

	def on_reset(self):
		"""
		Called at the beginning of each :meth:`reset`, before ``world.reset()``.
		Override for procedural generation.
		"""
		pass

	# ── PettingZoo interface ──

	def observation_space(self, agent_id):
		"""Return the observation space for the given agent."""
		return self._observation_spaces[agent_id]

	def action_space(self, agent_id):
		"""Return the action space for the given agent."""
		return self._action_spaces[agent_id]

	def step(self, actions):
		"""
		Apply actions for all agents, advance the simulation.

		Parameters
		----------
		actions : dict[str, np.ndarray]
			Mapping of agent_id to action array.

		Returns
		-------
		tuple
			``(observations, rewards, terminations, truncations, infos)``
			each as ``dict[str, ...]``.
		"""
		for agent_id, action in actions.items():
			agent = self._agents_by_id[agent_id]
			agent.force = np.asarray(action, dtype=np.float64)
		self.world.step(n_steps=self.n_substeps)
		observations = {}
		rewards = {}
		terminations = {}
		truncations = {}
		infos = {}
		for agent_id in self.agents:
			agent = self._agents_by_id[agent_id]
			observations[agent_id] = self._get_obs(agent)
			rewards[agent_id] = float(self.reward(agent_id, actions.get(agent_id)))
			terminations[agent_id] = bool(self.terminated(agent_id))
			truncations[agent_id] = bool(self.truncated(agent_id))
			infos[agent_id] = self.info(agent_id)
		return observations, rewards, terminations, truncations, infos

	def reset(self, seed=None, options=None):
		"""
		Reset the environment.

		Parameters
		----------
		seed : int or None, optional
			Random seed.
		options : dict or None, optional
			Additional options (unused by default).

		Returns
		-------
		tuple
			``(observations, infos)`` each as ``dict[str, ...]``.
		"""
		if seed is not None:
			np.random.seed(seed)
		self.agents = list(self.possible_agents)
		self.on_reset()
		self.world.reset()
		observations = {
			aid: self._get_obs(self._agents_by_id[aid])
			for aid in self.agents
		}
		infos = {aid: self.info(aid) for aid in self.agents}
		return observations, infos

	def render(self):
		"""
		Render the environment.

		Returns
		-------
		np.ndarray or None
			RGB image from the first agent's first camera if
			``render_mode="rgb_array"``, otherwise ``None``.
		"""
		if self.render_mode != "rgb_array":
			return None
		for agent in self._agents_by_id.values():
			cameras = agent.camera_observation
			if cameras:
				return next(iter(cameras.values()))
		return None

	def close(self):
		"""Unbuild the world and release resources."""
		if hasattr(self, 'world') and self.world._built:
			self.world.unbuild()

	# ── Internal helpers ──

	def _build_obs_space(self, agent):
		total = sum(
			dim for (dim,) in agent.sensor_observation_shape.values()
		)
		return gymnasium.spaces.Box(
			low=-np.inf, high=np.inf,
			shape=(total,), dtype=np.float64,
		)

	def _build_act_space(self, agent):
		n = len(list(agent.actuators))
		if n == 0:
			raise ValueError(
				f"Agent '{agent.name}' has no actuators. "
				f"Attach at least one actuator to define an action space."
			)
		return gymnasium.spaces.Box(
			low=-np.inf, high=np.inf,
			shape=(n,), dtype=np.float64,
		)

	def _get_obs(self, agent):
		values = list(agent.sensor_observation.values())
		return np.concatenate(values) if values else np.array([], dtype=np.float64)
