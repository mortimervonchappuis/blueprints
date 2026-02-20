from setuptools import setup, find_packages

with open('README.md', 'r') as file:
	long_description = file.read()

setup(name='mujoco-blueprints', 
      version='0.0.5', 
      packages=find_packages(), 
      author='Mortimer von Chappuis', 
      install_requires=['numpy>=2.0.0', 
      			'imageio>=2.0.0', 
      			'mujoco>=3.3.2', 
      			'tqdm>=4.0.0'], 
      description='Blueprints is a Mujoco interface providing many graph manipulation operations and easy runtime data access.', 
      long_description=long_description, 
      long_description_content_type='text/markdown', 
      url='https://mujoco-blueprints.readthedocs.io/en/latest/index.html', 
      download_url='https://github.com/mortimervonchappuis/mujoco_blueprints')