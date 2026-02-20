import blueprints as blue
import gymnasium as gym
import numpy as np
from ant import world, ant, hfield



world.build()



class Formicarium(gym.Env):
	def __init__(self, skip_frames: int = None):
		self.world, self.ant, self.hfield = world, ant, hfield
		#self._create_world()
		self._n_steps = skip_frames if skip_frames else 1
		# ACTION SPACE
		self.action_space = gym.spaces.Box(low=float('-inf'), high=float('inf'), shape=(self.ant.action_shape['force'],))
		# OBSERVATION SPACE
		self.observation_space = gym.spaces.Box(low=float('-inf'), 
							high=float('inf'), 
							shape=(sum(i for (i,) in self.ant.sensor_observation_shape.values()),))


	def _create_terrain(self, seed: int = None):
		if seed is not None:
			np.random.seed(seed)
		resolution = (50, 1000)
		heights = np.zeros(resolution)
		for frequency in range(1, 10):
			heights += 1/frequency * blue.perlin(resolution, frequency)
		return heights


	def _get_obs(self):
		pose = np.concatenate(list(self.ant.sensor_observation.values()))
		return {'pose': pose, 'pos': self.ant.pos}


	def _get_reward(self, action):
		vel_reward = self.ant.x_vel
		term_reward = 10 if self.ant.x > 45 else 0
		reward = vel_reward + term_reward
		cost = 0.5 * np.sqrt(np.sum(action**2))
		info = {'term': term_reward, 'vel': vel_reward, 'cost': cost}
		return reward - cost, info


	def reset(self, seed: int = None, options: dict = None):
		if seed is not None:
			np.random.seed(seed)
		self.hfield.terrain = self._create_terrain()
		self.world.reset()
		# RETURNS
		observation = self._get_obs()
		info = {}
		return observation, info


	def step(self, action):
		# APPLY ACTIONS
		self.ant.force = action
		# UPDATE ENV
		self._last_x = self.ant.x
		self.world.step(self._n_steps)
		# COMPUTE RETURNS
		truncation = self.ant.z < -10 # ANT HAS GLITCHED
		termination = self.ant.x > 45 # ANT HAS TERMINATED
		reward, info = self._get_reward(action)
		observation = self._get_obs()
		return observation, reward, termination, truncation, info

	def render(self):
		return self.ant.camera_observation



gym.register(id='Formicarium-v0',
	     entry_point=Formicarium,
	     max_episode_steps=10_000)


if __name__ == '__main__':
	import tqdm


	env = gym.make('Formicarium-v0')
	observation, info = env.reset(seed=42)
	bar = tqdm.tqdm(range(1000))
	for i in bar:
		action = env.action_space.sample()
		observation, reward, terminated, truncated, info = env.step(action)
		if terminated or truncated:
			observation, info = env.reset()
			print(terminated, truncated)
		bar.set_postfix(info)
	env.close()

