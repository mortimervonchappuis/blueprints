import numpy as np
from math import trunc, ceil
from itertools import product



def smoothstep(x):
	y = 3 * x**2 - 2 * x**3
	y[x <= 0] = 0.
	y[x >= 1] = 1.
	return y



def padding(x, axis):
	axis = tuple(slice(None if i == -1 else 1, -1 if i == -1 else None) for i in axis)
	return x[*axis,...]



def resize(img, shape):
	x_shape = img.shape
	y_shape = shape
	for x, y in zip(x_shape, y_shape):
		X = np.arange(x)[None,...]
		Y = np.arange(y)[...,None]
		mini = np.minimum((1 + X) * y / x, (1 + Y))
		maxi = np.maximum(     X  * y / x,      Y )
		M = np.maximum(0, mini - maxi)
		img = np.einsum('ij,j...->i...', M, img)
		img = np.moveaxis(img, 0, -1)
	return img



def perlin(resolution, frequency, periodic=False):
	ALPHA = 'abcdefghijklmnopqrstuvwx'
	min_width = int(ceil(min(resolution) / frequency))
	grid_size = tuple(int(ceil(x / min_width)) for x in resolution)
	cell_size = tuple(int(ceil(x / y)) for x, y in zip(resolution, grid_size))
	ndim      = len(grid_size)
	cell      = np.stack(np.meshgrid(*(np.linspace(-1, 1, n) for n in cell_size)), axis=-1)
	corners   = list(map(np.array, product(*((-1, 1) for _ in range(ndim)))))
	offsets   = np.stack(list(map(lambda x: cell - x, corners)), axis=-1)
	distances = np.sqrt(np.sum(offsets**2, axis=-2)) / 2
	factors   = 1 - smoothstep(distances)
	grads     = np.random.uniform(low=-1, high=1, size=(*(x + 1 for x in grid_size), ndim))
	grads     = grads / (1e-8 + np.sqrt(np.sum(grads**2, axis=-1)[...,None]))
	if periodic:
		for i, _ in enumerate(grid_size):
			grads[*(slice(None) for _ in range(i)), -1, ...] = grads[*(slice(None) for _ in range(i)),0,...]
	rule     = f'{ALPHA[:ndim].upper()}y,{ALPHA[:ndim]}yz->{ALPHA[:ndim].upper()}{ALPHA[:ndim]}z'
	products = np.einsum(rule, grads, offsets)
	correlations = np.stack([padding(products[...,i], corner) for i, corner in enumerate(corners)], axis=-1)
	rule     = f'{ALPHA[:ndim].upper()}{ALPHA[:ndim]}y,{ALPHA[:ndim]}y->{ALPHA[:ndim].upper()}{ALPHA[:ndim]}'
	heights  = np.einsum(rule, correlations, factors)
	heights  = np.moveaxis(heights, 0, 1)
	for _ in range(ndim):
		heights = np.concatenate(np.rollaxis(heights, axis=0), axis=ndim-1)
	heights  = np.moveaxis(heights, 0, 1)
	heights  = resize(heights, resolution)
	return heights
