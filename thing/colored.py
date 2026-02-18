import blueprints as blue
from blueprints import restrict
from . import base

import numpy as np
import inspect
import xml.etree.ElementTree as xml
from copy import copy
from collections import defaultdict



@blue.restrict
def gradient(*colors: list[str|int|list[int|float]|np.ndarray|blue.ColorType], 
	     n_steps: int):
	"""
	This function creates a color gradient in form of a list of colors, 
	which are interpolated between argument color.

	>>> blue.gradient('red', 'yellow', n_steps=6)
	[Red[#FF0000], Red[#F52900], Orange[#EB5200], Orange[#E07A00], Yellow[#D6A300], Yellow[#CCCC00]]


	Parameters
	----------
	colors : list
		A list of colors given in any of the formats specified by :class:`Color`.
	n_steps : int
		The number of steps taken from the first to the last color.

	Returns
	-------
	list
		List of colors interpolated in n_steps along all given colors.
	"""
	n_colors = len(colors)
	if n_colors < 2:
		raise ValueError(f'At least two colors must be given. Got {n_colors} instead.')
	if n_steps < 1:
		raise ValueError(f'Argument n_steps must be larger than 0, got {n_steps} instead.')
	if n_colors > n_steps:
		raise ValueErrur(f'There must be more intermediate steps then colors')
	ratio = (n_colors-1)/(n_steps - 1)
	func  = lambda i, j: max(0, 1 - abs(i * ratio - j))
	rate  = np.array([[[func(i, j)] for j in range(n_colors)]\
					for i in range(n_steps)])
	rgbs  = np.array([Color(color).rgb for color in colors])[None,...]
	grad  = np.sum(rgbs * rate, axis=1)
	grad  = np.maximum(0, np.minimum(1, grad))
	return list(map(Color, np.rollaxis(grad, axis=0)))



class Color(blue.ColorType):

	"""
	The color descriptor can be one of the following:

	 * string containing the colors name: ``'orange'``
	 * string containing the colors hex code as RGB: ``'#FFFF00'``
	 * string containing the colors hex code as RGBA: ``'#FFFF0080'``
	 * integer specifying the colors hex code as RGB ``0xFFFF00``
	 * list containing the colors components as RGB: ``[1, 1, 0]``
	 * list containing the colors components as RGBA: ``[1, 1, 0, 0.5]``
	 * np.array containing the colors components as RGB: ``[1, 1, 0]``
	 * np.array containing the colors components as RGBA: ``[1, 1, 0, 0.5]``
	 * Color object.
	"""

	@restrict
	def __init__(self, 
		     descriptor: str|int|list[int|float]|np.ndarray|blue.ColorType, 
		     opacity: int|float|None = None):
		if isinstance(opacity, (int, float)) and not 0 <= opacity <= 1:
			raise ValueError(f'Opacity must either be None or in range [0, 1], got {opacity} instead.')
		try:
			if isinstance(descriptor, str):
				rgba = self._from_string(descriptor, opacity)
			elif isinstance(descriptor, int):
				rgba = self._from_int(descriptor, opacity)
			elif isinstance(descriptor, list):
				rgba = self._from_list(descriptor, opacity)
			elif isinstance(descriptor, np.ndarray):
				rgba = self._from_array(descriptor, opacity)
			elif isinstance(descriptor, blue.ColorType):
				rgba = descriptor.rgba
			else:
				raise Exception('This line should never be reached. Please reboot your local universe.')
		except ValueError as error:
			raise error from None
		self.rgba = rgba


	def __repr__(self):
		"""
		The representations contains the name of the color and the RGB hex value.

		Returns
		-------
		str
		"""
		return f'{self.name.title()}[#{self._hex_from_array(self.rgb).upper()}]'


	@property
	def name(self) -> str:
		"""
		The name is set to be the closest name among default colors.

		Returns
		-------
		str
		"""
		return min(self._COLORS.items(), key=lambda x: np.linalg.norm(self.rgb - x[1]))[0]


	@property
	def rgb(self) -> np.ndarray:
		"""
		The values are synchronized with :attr:`rgba`, :attr:`red`,:attr:`green` and :attr:`blue` .

		Returns
		-------
		np.ndarray
			The color array without opacity.
		"""
		return self._rgba[:3]


	@rgb.setter
	@restrict
	def rgb(self, rgb: list[int|float]|np.ndarray) -> None:
		"""
		The values are synchronized with :attr:`rgba`, :attr:`red`,:attr:`green` and :attr:`blue` .

		Parameters
		----------
		rgb : list[int | float] | np.ndarray
			The color array without opacity.

		Raises
		------
		ValueError
			If any component is not in the range [0, 1] an error is raised.
		"""
		rgb = np.array(rgb, dtype=np.float32)
		if not np.all(0 <= rgb) and np.all(rgb <= 1):
			raise ValueError(f'All color components must be in range [0, 1], got {rgb}.')
		self._rgba[:3] = rgb


	@property
	def rgba(self) -> np.ndarray:
		"""
		The values are synchronized with :attr:`rgb`, :attr:`red`,:attr:`green`, :attr:`blue` and :attr:`opacity`.

		Returns
		-------
		np.ndarray
			The color array with opacity.
		"""
		return self._rgba


	@rgba.setter
	@restrict
	def rgba(self, rgba: list[int|float]|np.ndarray) -> None:
		"""
		The values are synchronized with :attr:`rgb`, :attr:`red`,:attr:`green`, :attr:`blue` and :attr:`opacity`.

		Parameters
		----------
		rgba : list[int | float] | np.ndarray
			The color array with opacity.

		Raises
		------
		ValueError
			If any component is not in the range [0, 1] an error is raised.
		"""
		rgba = np.array(rgba, dtype=np.float32)
		if not np.all(0 <= rgba) and np.all(rgba <= 1):
			raise ValueError(f'All color components must be in range [0, 1], got {rgba}.')
		self._rgba = rgba


	@property
	def red(self) -> float|None:
		"""
		The value is synchronized with :attr:`rgb` and :attr:`rgba`.

		Returns
		-------
		float | None
			The red component of the color array.
		"""
		return float(self._rgba[0])


	@red.setter
	@restrict
	def red(self, red: int|float|np.int32|np.int64|np.float32|np.float64) -> None:
		"""
		The value is synchronized with :attr:`rgb` and :attr:`rgba`.

		Parameters
		----------
		red : int | float | np.int32 | np.int64 | np.float32 | np.float64
			The red component of the color array.

		Raises
		------
		ValueError
			If the red component is not in the range [0, 1] an error is raised.
		"""
		if not 0 <= red <= 1:
			raise ValueError(f'All color components must be in range [0, 1], got {red}.')
		self._rgba[0] = float(red)


	@property
	def green(self) -> float|None:
		"""
		The value is synchronized with :attr:`rgb` and :attr:`rgba`.
		
		Returns
		-------
		float | None
			The green component of the color array.
		"""
		return float(self._rgba[1])


	@green.setter
	@restrict
	def green(self, green: int|float|np.int32|np.int64|np.float32|np.float64) -> None:
		"""
		The value is synchronized with :attr:`rgb` and :attr:`rgba`.
		
		Parameters
		----------
		green : int | float | np.int32 | np.int64 | np.float32 | np.float64
			The green component of the color array.

		Raises
		------
		ValueError
			If the green component is not in the range [0, 1] an error is raised.
		"""
		if not 0 <= green <= 1:
			raise ValueError(f'All color components must be in range [0, 1], got {green}.')
		self._rgba[1] = float(green)


	@property
	def blue(self) -> float|None:
		"""
		The value is synchronized with :attr:`rgb` and :attr:`rgba`.
		
		Returns
		-------
		float | None
			The blue component of the color array.
		"""
		return float(self._rgba[2])


	@blue.setter
	@restrict
	def blue(self, blue: int|float|np.int32|np.int64|np.float32|np.float64) -> None:
		"""
		The value is synchronized with :attr:`rgb` and :attr:`rgba`.
		
		Parameters
		----------
		blue : int | float | np.int32 | np.int64 | np.float32 | np.float64
			The blue component of the color array.

		Raises
		------
		ValueError
			If the blue component is not in the range [0, 1] an error is raised.
		"""
		if not 0 <= blue <= 1:
			raise ValueError(f'All color components must be in range [0, 1], got {blue}.')
		self._rgba[2] = float(blue)


	@property
	def opacity(self) -> int|float:
		"""
		The value is synchronized with :attr:`rgba`.
		
		Returns
		-------
		int | float
			The opacity component of the color array.
		"""
		return self._rgba[3]


	@opacity.setter
	@restrict
	def opacity(self, opacity: int|float) -> None:
		"""
		The value is synchronized with :attr:`rgba`.
		
		Parameters
		----------
		opacity : int | float
			The opacity component of the color array.

		Raises
		------
		ValueError
			If the opacity component is not in the range [0, 1] an error is raised.
		"""
		if not 0 <= opacity <= 1:
			raise ValueError(f'Opacity must be in range [0, 1], got {opacity}.')
		self._rgba[3] = float(opacity)



	# INIT METHODS

	@classmethod
	def _from_string(cls, descriptor, opacity):
		if descriptor.startswith('#'):
			rgba = cls._from_hex_string(descriptor, opacity)
		else:
			rgba = cls._from_name_string(descriptor, opacity)
		return rgba


	@classmethod
	def _from_int(cls, descriptor, opacity):
		red, green, blue = cls._rgb_from_hex(descriptor)
		opacity = opacity if opacity is not None else 1.0
		return [red/0xff, green/0xff, blue/0xff, opacity]


	@classmethod
	def _from_list(cls, descriptor, opacity):
		if len(descriptor) == 3:
			opacity = opacity if opacity is not None else 1.0
			rgba = descriptor + [opacity]
		elif len(descriptor) == 4:
			if opacity is None:
				rgba = descriptor
			else:
				opacity = opacity if opacity is not None else 1.0
				rgba = descriptor[:3] + [opacity]
		else:
			raise ValueError(f'The allowed lengths for descriptor as list is either 3 or 4, got {len(list)} instead.')
		return rgba


	@classmethod
	def _from_array(cls, descriptor, opacity):
		if descriptor.shape == (3,):
			opacity = opacity if opacity is not None else 1.0
			rgba = list(map(float, descriptor)) + [opacity]
		elif descriptor.shape == (4,):
			if opacity is None:
				rgba = list(map(float, descriptor))
			else:
				opacity = opacity if opacity is not None else 1.0
				rgba = list(map(float, descriptor[:3])) + [opacity]
		return rgba


	@classmethod
	def _from_hex_string(cls, descriptor, opacity):
		error = ValueError(f"""The string descriptor of a rgb or rgba hex code is not formated correctly. The received argument was {descriptor}.

RGB may be formated like this: 
'#E73D8A'
'#e73d8a'
RGBA may be formated like this:
'#E73D8AFF'
'#e73d8aff'
""")
		if len(descriptor) != 7 and len(descriptor) != 9:
			raise error
		NUMERALS = '0123456789abcdefABCDEF'
		if any(c not in NUMERALS for c in descriptor[1:]):
			raise error
		if len(descriptor) == 7:
			red, green, blue = cls._rgb_from_hex(int(descriptor[1:], base=16))
			opacity = opacity if opacity is not None else 1.0
			rgba = [red/0xff, green/0xff, blue/0xff, opacity]
		elif len(descriptor) == 9:
			red, green, blue, opacity = cls._rgba_from_hex(int(descriptor[1:], base=16))
			rgba = [red/0xff, green/0xff, blue/0xff, opacity/0xff]
		return rgba

	@classmethod
	def _from_name_string(cls, descriptor, opacity):
		if descriptor not in cls._COLORS:
			raise ValueError(f"Color expected descriptor to be {', '.join(cls._COLORS.keys())}, got {descriptor} instead.")
		opacity = opacity if opacity is not None else 1.0
		return cls._COLORS[descriptor] + [opacity]

	# CONVERSION METHODS

	@classmethod
	def _hex_from_array(cls, array):
		return ''.join(map(lambda n: hex(int(n))[2:].zfill(2), np.round(array * 0xff)))


	@classmethod
	def _rgb_from_hex(cls, n):
		"""
		Helper function to get the rgb from n

		Parameters
		----------
		n : int
			The int gets interpreted as rgb in hex

		Returns
		-------
		int, int, int
		"""
		red, green, blue = map(lambda p: cls._from_hex_partition(n, p), [2, 1, 0])
		return red, green, blue


	@classmethod
	def _rgba_from_hex(cls, n):
		"""
		Helper function to get the rgba from n

		Parameters
		----------
		n : int
			The int gets interpreted as rgba in hex

		Returns
		-------
		int, int, int, int
		"""
		red, green, blue, opacity = map(lambda p: cls._from_hex_partition(n, p), [3, 2, 1, 0])
		return red, green, blue, opacity


	@staticmethod
	def _from_hex_partition(n, p) -> int:
		"""
		Helper function to get the p-th partition of the hex int (index is read from right to left).

		Returns
		-------
		int
		"""
		return (n >> (p * 8)) & 0xff



class ColoredThing(blue.ColoredThingType, base.BaseThing):

	"""
	This class is used to implement color. The color components are stored in the :attr:`rgba` property. 
	Those components are also accessible as individual properties :attr:`red`, :attr:`green`, :attr:`blue` 
	and :attr:`opacity`. All components of the color array must be in the range [0, 1].

	.. note::
		The transparency of :class:`ColoredThing` is modified via :attr:`opacity`, and not with 
		``alpha``, since this property is reserved for :attr:`blueprints.thing.moveable.MoveableThing.alpha`!
	
	Parameters
	----------
	rgb : list[int | float] | np.ndarray | None, optional
		The color array without opacity.
	rgba : list[int | float] | np.ndarray | None, optional
		The color array with opacity.
	**kwargs
		Keyward arguments are passed to ``super().__init__``.
	"""

	@restrict
	def __init__(self, 
		     color='white', 
		     opacity=None, 
		     **kwargs):
		"""
		Parameters
		----------
		rgb : list[int | float] | np.ndarray | None, optional
			The color array without opacity.
		rgba : list[int | float] | np.ndarray | None, optional
			The color array with opacity.
		**kwargs
			Keyward arguments are passed to ``super().__init__``.
		"""
		self.color = color
		if opacity:
			self.opacity = opacity
		super().__init__(**kwargs)


	@property
	def color(self) -> blue.ColorType:
		"""
		The :class:`Color` contains all attributes of the :class:`ColoredThing` color components.

		Returns
		-------
		Color
		"""
		return self._color


	@color.setter
	@restrict
	def color(self, color: str|int|list[int|float]|np.ndarray|blue.ColorType|None) -> None:
		"""
		The color attribute is of the type:class:`Color` which contains all attributes of the 
		:class:`ColoredThing` color components.

		Parameters
		----------
		color : str|int|list[int|float]|np.ndarray|blue.ColorType
			The color descriptor can be one of the following:

			 * A string containing the colors name, for example ``'orange'``
			 * A string containing the colors hex code as RGB, for example ``'#FFFF00'``
			 * A string containing the colors hex code as RGBA, for example ``'#FFFF0080'``
			 * A list containing the colors components as RGB, for example ``[1, 1, 0]``
			 * A list containing the colors components as RGBA, for example ``[1, 1, 0, 0.5]``
			 * A np.array containing the colors components as RGB, for example ``[1, 1, 0]``
			 * A np.array containing the colors components as RGB, for example ``[1, 1, 0, 0.5]``
			 * A Color object.
		"""
		if color is not None:
			self._color = Color(color)
		else:
			self._color = None



	@property
	def rgb(self) -> np.ndarray:
		"""
		The values are synchronized with :attr:`rgba`, :attr:`red`,:attr:`green` and :attr:`blue` .

		Returns
		-------
		np.ndarray
			The color array without opacity.
		"""
		return self.color.rgb


	@rgb.setter
	@restrict
	def rgb(self, rgb: list[int|float]|np.ndarray) -> None:
		"""
		The values are synchronized with :attr:`rgba`, :attr:`red`,:attr:`green` and :attr:`blue` .

		Parameters
		----------
		rgb : list[int | float] | np.ndarray
			The color array without opacity.

		Raises
		------
		ValueError
			If any component is not in the range [0, 1] an error is raised.
		"""
		self.color.rgb = rgb


	@property
	def rgba(self) -> np.ndarray|None:
		"""
		The values are synchronized with :attr:`rgb`, :attr:`red`,:attr:`green`, :attr:`blue` and :attr:`opacity`.

		Returns
		-------
		np.ndarray
			The color array with opacity.
		"""
		if self.color is not None:
			return self.color.rgba
		else:
			return None


	@rgba.setter
	@restrict
	def rgba(self, rgba: list[int|float]|np.ndarray) -> None:
		"""
		The values are synchronized with :attr:`rgb`, :attr:`red`,:attr:`green`, :attr:`blue` and :attr:`opacity`.

		Parameters
		----------
		rgba : list[int | float] | np.ndarray
			The color array with opacity.

		Raises
		------
		ValueError
			If any component is not in the range [0, 1] an error is raised.
		"""
		if self.color is None:
			self.color = Color([0, 0, 0])
		self.color.rgba = rgba


	@property
	def red(self) -> float|None:
		"""
		The value is synchronized with :attr:`rgb` and :attr:`rgba`.

		Returns
		-------
		float | None
			The red component of the color array.
		"""
		return self.color.red


	@red.setter
	@restrict
	def red(self, red: int|float|np.int32|np.int64|np.float32|np.float64) -> None:
		"""
		The value is synchronized with :attr:`rgb` and :attr:`rgba`.

		Parameters
		----------
		red : int | float | np.int32 | np.int64 | np.float32 | np.float64
			The red component of the color array.

		Raises
		------
		ValueError
			If the red component is not in the range [0, 1] an error is raised.
		"""
		if self.color is None:
			self.color = Color([0, 0, 0])
		self.color.red = red


	@property
	def green(self) -> float|None:
		"""
		The value is synchronized with :attr:`rgb` and :attr:`rgba`.
		
		Returns
		-------
		float | None
			The green component of the color array.
		"""
		return self.color.green


	@green.setter
	@restrict
	def green(self, green: int|float|np.int32|np.int64|np.float32|np.float64) -> None:
		"""
		The value is synchronized with :attr:`rgb` and :attr:`rgba`.
		
		Parameters
		----------
		green : int | float | np.int32 | np.int64 | np.float32 | np.float64
			The green component of the color array.

		Raises
		------
		ValueError
			If the green component is not in the range [0, 1] an error is raised.
		"""
		if self.color is None:
			self.color = Color([0, 0, 0])
		self.color.green = green


	@property
	def blue(self) -> float|None:
		"""
		The value is synchronized with :attr:`rgb` and :attr:`rgba`.
		
		Returns
		-------
		float | None
			The blue component of the color array.
		"""
		return self.color.blue


	@blue.setter
	@restrict
	def blue(self, blue: int|float|np.int32|np.int64|np.float32|np.float64) -> None:
		"""
		The value is synchronized with :attr:`rgb` and :attr:`rgba`.
		
		Parameters
		----------
		blue : int | float | np.int32 | np.int64 | np.float32 | np.float64
			The blue component of the color array.

		Raises
		------
		ValueError
			If the blue component is not in the range [0, 1] an error is raised.
		"""
		if self.color is None:
			self.color = Color([0, 0, 0])
		self.color.blue = blue


	@property
	def opacity(self) -> int|float:
		"""
		The value is synchronized with :attr:`rgba`.
		
		Returns
		-------
		int | float
			The opacity component of the color array.
		"""
		return self.color.opacity


	@opacity.setter
	@restrict
	def opacity(self, opacity: int|float) -> None:
		"""
		The value is synchronized with :attr:`rgba`.
		
		Parameters
		----------
		opacity : int | float
			The opacity component of the color array.

		Raises
		------
		ValueError
			If the opacity component is not in the range [0, 1] an error is raised.
		"""
		if self.color is None:
			self.color = Color([0, 0, 0])
		self.color.opacity = opacity
