import blueprints as blue


torso = blue.Body(name='torso')
sphere = blue.geoms.Sphere(radius=0.3)
torso.attach(sphere)

hip_geom = blue.geoms.Capsule.from_points([0, 0, 0], [0, 0.3, 0.0], radius=0.08)
hip = blue.Body(name='hip', geoms=hip_geom)

hips = (hip.rotate(gamma=blue.TAU/4 * i) for i in range(4))
torso.attach(*hips)

torso.all.color = 'white'
#torso.all.color = 'grey'
#torso.view()

uleg_geom = blue.geoms.Capsule.from_points([0, 0, 0], [0, 0.3, 0], radius=0.08)
uleg_hinge = blue.joints.Hinge(axis=[0, 0, 1], range=[-blue.TAU/12, blue.TAU/12])
uleg = blue.Body(name='upper_leg', geoms=uleg_geom, joints=uleg_hinge, y=0.3)

lleg_geom = blue.geoms.Capsule.from_points([0, 0, 0], [0,0.2,-0.6], radius=0.08)
lleg_hinge = blue.joints.Hinge(axis=[-1, 0, 0], range=[-blue.TAU/12, blue.TAU/8])
lleg = blue.Body(name='lower_leg', geoms=lleg_geom, joints=lleg_hinge, y=0.3)


uleg.attach(lleg)

#uleg.all.color = 'white'
uleg.all.color = 'black'
#uleg.view()
#print(repr(torso.bodies))
#print(torso.bodies.attach)

torso.bodies.attach(uleg)

hinge = blue.joints.Hinge(axis=[1, 0, 0], range=[-blue.TAU/12, blue.TAU/8])
torso.bodies.bodies.attach(hinge)

motor = blue.actuators.Motor(ctrlrange=[-1, 1], gear=[150])
torso.all.joints.attach(motor)

torso.all.joints.armature = 1 
torso.all.joints.damping = 1

torso.all.geoms.density = 5.0
torso.all.margin = 0.01
torso.all.geoms.sliding_friction = 1
torso.all.geoms.torsional_friction = 0.5
torso.all.geoms.rolling_friction = 0.5

#torso.all.joints.attach(blue.sensors.JointVel(), blue.sensors.JointPos())
torso.all.joints.attach(blue.sensors.JointPos())
torso.all.joints.attach(blue.sensors.JointVel())

torso.all.color = 'white'
#torso.all.color = 'grey'
#torso.view()

foot = blue.sites.Sphere(radius=0.1, sensors=blue.sensors.Touch())
foot = foot.locate(lleg_geom.tail)
torso.all.bodies['lower_leg'].attach(foot)

ant = blue.Agent(bodies=torso.rotate(gamma=blue.TAU/8), joints=blue.joints.Free())
ant.all.color = 'black'

camera = blue.Camera(x=-3, z=2).looking(ant)
ant.attach(camera)


#ant.view()

#print(ant.observation_shape)
#print(ant.action_shape)



#heights = blue.perlin((50, 1000), frequency=2)
#blue.geoms.HField(terrain=heights, x_length=100, y_length=5).view()
#print(hfield.asset.x_length)
#hfield.view()

import numpy as np
resolution = (50, 1000)
heights = np.zeros(resolution)
for frequency in range(1, 10):
	heights += 1/frequency * blue.perlin(resolution, frequency)


hfield = blue.geoms.HField(terrain=heights, x_length=100, y_length=5, z_length=2, name='ground')

#hfield.view()

hfield.sliding_friction = 2
hfield.torsional_friction = 0.5
hfield.rolling_friction = 5

#print(repr(hfield))


wall_long  = blue.geoms.Box(x_length=101, y_length=0.5, z_length=4, z=1.5, name='wall')
wall_short = blue.geoms.Box(x_length=0.5, y_length=5.0, z_length=4, z=1.5, name='wall')


world = blue.World(timestep=0.01)
world.attach(hfield)

#print(repr(world.geoms['ground'][0]))
#exit()
#world.attach(hfield.lattice([[100, 0, 0], [0, 5, 0]], [5, 5]))
world.attach(wall_short.shift(x=50.25), wall_short.shift(x=-50.25), 
	wall_long.shift(y=2.75), wall_long.shift(y=-2.75))

#print(world.all.geoms['wall'].color)
world.all.geoms['wall'].color = 'white'
world.all.geoms['wall'].opacity = 0.1



#world.view()
#world.unbuild()




sand_tex = blue.texture.Plane(builtin='flat', 
			      mark='random', 
			      random=0.3, 
			      color_1='#CC9977',
			      color_mark='#BB8866', 
			      width=1000, 
			      height=50)
sand_mat = blue.Material(texture=sand_tex)
hfield.material = sand_mat

sky = blue.texture.Skybox(builtin='gradient', 
			  color_1='#64B5F6',
			  color_2='#0277BD', 
			  width=1000, 
			  height=1000)
world.texture = sky
world.attach(blue.Light(z=100))
world.attach(blue.geoms.Plane(color='#CC9977'))
#print(world.geoms.ID, hfield.ID)

world.attach(ant.locate(x=-48, z=1.5))

ant = world.agents[0]

hfield.terrain = heights

if __name__ == '__main__':
	world.view()
#world.attach(ant, blue.geoms.Plane())

#world.build()
#world.step(n_steps=200)
#print(world.all.sensors.observation)

#world.build()
#print(ant.all.bodies)
#print(world.data.cfrc_ext.shape)
#for i in range(100000):
#	world.step()
#	if not np.all(world.data.cfrc_ext == np.zeros((15, 6))):
#		print(i)
#		print(world.data.cfrc_ext)
#world.reset()
#print(ant.observation_shape)
#print()
#print(ant.action_shape)
#print()
#print(ant.observation)

#world.view()
#exit()