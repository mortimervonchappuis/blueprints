import numpy as np
import gymnasium
import blueprints as blue


class Env(gymnasium.Env):
	"""
	Base class for creating Gymnasium environments with mujoco_blueprints.

	Subclass and implement:

	- :meth:`setup` *(required)* — create ``self.world`` with at least one :class:`Agent <blueprints.agent.Agent>`
	- :meth:`reward` *(required)* — compute scalar reward for current state and action
	- :meth:`terminated` *(optional)* — episode success/failure condition
	- :meth:`truncated` *(optional)* — episode safety/timeout cutoff
	- :meth:`info` *(optional)* — extra info dict for logging
	- :meth:`on_reset` *(optional)* — called each reset before ``world.reset()``

	Example::

		from blueprints.rl import Env
		import blueprints as blue

		class MyEnv(Env):
			n_substeps = 4

			def setup(self):
				self.world = blue.World()
				agent = blue.Agent(name='robot', pos=[0, 0, 1])
				geom = blue.geoms.Sphere(size=0.1)
				joint = blue.joints.Hinge(axis=[0, 0, 1])
				actuator = blue.actuators.Motor(gear=[100])
				sensor = blue.sensors.JointPos()
				joint.attach(actuator, sensor)
				agent.attach(geom, joint)
				self.world.attach(agent)

			def reward(self, action):
				return self.agent.x_vel - 0.01 * sum(action**2)

	Attributes
	----------
	n_substeps : int
		Number of MuJoCo simulation steps per :meth:`step` call (frame skip). Default: 1.
	agent_name : str or None
		Name of the :class:`Agent <blueprints.agent.Agent>` to use. If ``None``
		and the world contains exactly one agent, it is auto-detected.
	include_cameras : bool
		If ``True``, camera images are included in the observation space
		(switches to a ``gymnasium.spaces.Dict`` space). Default: ``False``.
	render_camera : str or None
		Name of the camera used by :meth:`render`. If ``None``, the first
		camera is used.
	"""

	n_substeps      = 1
	agent_name      = None
	include_cameras = False
	render_camera   = None

	def __init__(self, render_mode=None, **kwargs):
		self.render_mode = render_mode
		self.setup()
		if not hasattr(self, 'world') or not isinstance(self.world, blue.World):
			raise AttributeError(
				"setup() must set self.world to a blueprints.World instance."
			)
		if not self.world._built:
			self.world.build()
		self.agent = self._resolve_agent()
		self.observation_space = self._build_observation_space()
		self.action_space = self._build_action_space()

	# ── User-implemented methods ──

	def setup(self):
		"""
		Create ``self.world`` with at least one :class:`Agent <blueprints.agent.Agent>`.
		The base class calls ``world.build()`` after this method returns.

		Raises
		------
		NotImplementedError
			If not overridden by the subclass.
		"""
		raise NotImplementedError(
			"Subclasses must implement setup() and set self.world."
		)

	def reward(self, action):
		"""
		Compute the scalar reward for the current state and the action taken.

		Parameters
		----------
		action : np.ndarray
			The action that was applied this step.

		Returns
		-------
		float

		Raises
		------
		NotImplementedError
			If not overridden by the subclass.
		"""
		raise NotImplementedError(
			"Subclasses must implement reward(action)."
		)

	def terminated(self):
		"""
		Whether the episode has ended due to a terminal condition (success or failure).

		Returns
		-------
		bool
		"""
		return False

	def truncated(self):
		"""
		Whether the episode has been cut short (safety, timeout).

		Returns
		-------
		bool
		"""
		return False

	def info(self):
		"""
		Extra information dict returned alongside observations.

		Returns
		-------
		dict
		"""
		return {}

	def on_reset(self):
		"""
		Called at the beginning of each :meth:`reset`, before ``world.reset()``.
		Override this for procedural generation (e.g. randomising terrain).
		"""
		pass

	# ── Gymnasium interface ──

	def step(self, action):
		"""
		Apply an action, advance the simulation, and return the outcome.

		Parameters
		----------
		action : np.ndarray
			Force vector for the agent's actuators.

		Returns
		-------
		tuple
			``(observation, reward, terminated, truncated, info)``
		"""
		action = np.asarray(action, dtype=np.float64)
		self.agent.force = action
		self.world.step(n_steps=self.n_substeps)
		obs  = self._get_obs()
		rew  = float(self.reward(action))
		term = bool(self.terminated())
		trunc = bool(self.truncated())
		inf  = self.info()
		return obs, rew, term, trunc, inf

	def reset(self, seed=None, options=None):
		"""
		Reset the environment to its initial state.

		Parameters
		----------
		seed : int or None, optional
			Random seed for reproducibility.
		options : dict or None, optional
			Additional reset options (unused by default).

		Returns
		-------
		tuple
			``(observation, info)``
		"""
		super().reset(seed=seed)
		self.on_reset()
		self.world.reset()
		return self._get_obs(), self.info()

	def render(self):
		"""
		Render the environment.

		Returns
		-------
		np.ndarray or None
			RGB image if ``render_mode="rgb_array"`` and a camera exists,
			otherwise ``None``.
		"""
		if self.render_mode != "rgb_array":
			return None
		cameras = self.agent.camera_observation
		if not cameras:
			return None
		if self.render_camera is not None:
			for key, img in cameras.items():
				if self.render_camera in key:
					return img
		return next(iter(cameras.values()))

	def close(self):
		"""Unbuild the world and release resources."""
		if hasattr(self, 'world') and self.world._built:
			self.world.unbuild()

	# ── Internal helpers ──

	def _resolve_agent(self):
		"""Find the agent to use based on agent_name or auto-detect."""
		if hasattr(self, 'agent') and isinstance(self.agent, blue.Agent):
			return self.agent
		agents = list(self.world.agents)
		if len(agents) == 0:
			raise ValueError(
				"No Agent found in the World. "
				"Attach at least one blueprints.Agent."
			)
		if self.agent_name is not None:
			for a in agents:
				if a.name == self.agent_name:
					return a
			names = [a.name for a in agents]
			raise ValueError(
				f"Agent '{self.agent_name}' not found. "
				f"Available agents: {names}"
			)
		if len(agents) > 1:
			names = [a.name for a in agents]
			raise ValueError(
				f"World has {len(agents)} agents: {names}. "
				f"Set agent_name or assign self.agent in setup()."
			)
		return agents[0]

	def _build_observation_space(self):
		"""Construct the observation space from agent shapes."""
		if not self.include_cameras:
			total = sum(
				dim for (dim,) in
				self.agent.sensor_observation_shape.values()
			)
			return gymnasium.spaces.Box(
				low=-np.inf, high=np.inf,
				shape=(total,), dtype=np.float64,
			)
		spaces = {}
		total = sum(
			dim for (dim,) in
			self.agent.sensor_observation_shape.values()
		)
		spaces["sensors"] = gymnasium.spaces.Box(
			low=-np.inf, high=np.inf,
			shape=(total,), dtype=np.float64,
		)
		for name, shape in self.agent.camera_observation_shape.items():
			spaces[name] = gymnasium.spaces.Box(
				low=0, high=255,
				shape=(*shape, 3), dtype=np.uint8,
			)
		return gymnasium.spaces.Dict(spaces)

	def _build_action_space(self):
		"""Construct the action space from agent actuator count."""
		n_actuators = len(list(self.agent.actuators))
		if n_actuators == 0:
			raise ValueError(
				"Agent has no actuators. "
				"Attach at least one actuator to define an action space."
			)
		return gymnasium.spaces.Box(
			low=-np.inf, high=np.inf,
			shape=(n_actuators,), dtype=np.float64,
		)

	def _get_obs(self):
		"""Collect current observation from the agent."""
		sensor_obs = self.agent.sensor_observation
		values = list(sensor_obs.values())
		flat = np.concatenate(values) if values else np.array([], dtype=np.float64)
		if not self.include_cameras:
			return flat
		obs = {"sensors": flat}
		obs.update(self.agent.camera_observation)
		return obs
