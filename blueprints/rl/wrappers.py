"""
Gymnasium wrappers for mujoco_blueprints RL environments.

These wrap any ``gymnasium.Env`` (including :class:`blueprints.rl.Env`)
to transform observations or actions::

	from blueprints.rl import Env
	from blueprints.rl.wrappers import FlattenObservation, NormalizeObservation

	env = MyEnv()
	env = FlattenObservation(env)
	env = NormalizeObservation(env)
"""

import numpy as np
import gymnasium


class FlattenObservation(gymnasium.ObservationWrapper):
	"""
	Flatten a Dict observation space into a single Box.

	Camera images (uint8) are excluded — only float entries are concatenated.
	If the observation is already a Box, this wrapper is a no-op.
	"""

	def __init__(self, env):
		super().__init__(env)
		if isinstance(env.observation_space, gymnasium.spaces.Dict):
			total = 0
			self._keys = []
			for key, space in env.observation_space.spaces.items():
				if space.dtype == np.uint8:
					continue
				self._keys.append(key)
				total += int(np.prod(space.shape))
			self.observation_space = gymnasium.spaces.Box(
				low=-np.inf, high=np.inf,
				shape=(total,), dtype=np.float64,
			)
		else:
			self._keys = None

	def observation(self, obs):
		if self._keys is None:
			return obs
		parts = [np.asarray(obs[k], dtype=np.float64).ravel() for k in self._keys]
		return np.concatenate(parts) if parts else np.array([], dtype=np.float64)


class NormalizeObservation(gymnasium.Wrapper):
	"""
	Running mean/std normalization of observations.

	Tracks statistics online during training and normalizes
	``obs = (obs - mean) / (std + epsilon)``.

	Parameters
	----------
	env : gymnasium.Env
		The environment to wrap.
	epsilon : float
		Small constant to avoid division by zero. Default: 1e-8.
	"""

	def __init__(self, env, epsilon=1e-8):
		super().__init__(env)
		shape = env.observation_space.shape
		self._mean = np.zeros(shape, dtype=np.float64)
		self._var = np.ones(shape, dtype=np.float64)
		self._count = 0
		self._epsilon = epsilon

	def _update(self, obs):
		self._count += 1
		delta = obs - self._mean
		self._mean += delta / self._count
		delta2 = obs - self._mean
		self._var += (delta * delta2 - self._var) / self._count

	def _normalize(self, obs):
		return (obs - self._mean) / (np.sqrt(self._var) + self._epsilon)

	def step(self, action):
		obs, rew, term, trunc, info = self.env.step(action)
		self._update(obs)
		return self._normalize(obs), rew, term, trunc, info

	def reset(self, **kwargs):
		obs, info = self.env.reset(**kwargs)
		self._update(obs)
		return self._normalize(obs), info


class ActionClip(gymnasium.ActionWrapper):
	"""
	Clip actions to a specified range before passing to the environment.

	Parameters
	----------
	env : gymnasium.Env
		The environment to wrap.
	low : float
		Lower bound for clipping. Default: -1.0.
	high : float
		Upper bound for clipping. Default: 1.0.
	"""

	def __init__(self, env, low=-1.0, high=1.0):
		super().__init__(env)
		self._low = low
		self._high = high
		self.action_space = gymnasium.spaces.Box(
			low=low, high=high,
			shape=env.action_space.shape,
			dtype=env.action_space.dtype,
		)

	def action(self, action):
		return np.clip(action, self._low, self._high)


class ActionScale(gymnasium.ActionWrapper):
	"""
	Rescale actions from [-1, 1] to a target range.

	Useful when your policy outputs in [-1, 1] but the environment
	expects a different range.

	Parameters
	----------
	env : gymnasium.Env
		The environment to wrap.
	low : float
		Lower bound of the target range.
	high : float
		Upper bound of the target range.
	"""

	def __init__(self, env, low, high):
		super().__init__(env)
		self._low = float(low)
		self._high = float(high)
		self.action_space = gymnasium.spaces.Box(
			low=-1.0, high=1.0,
			shape=env.action_space.shape,
			dtype=env.action_space.dtype,
		)

	def action(self, action):
		action = np.asarray(action, dtype=np.float64)
		return self._low + (action + 1.0) * 0.5 * (self._high - self._low)
