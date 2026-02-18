"""
The Typechecker class is used to restrict the argument types of functions and methods. 
If a function or method is decoraded with TypeChecker.restrict its arguments and returned 
values are checked against the type hints, if provided and a TypeError is raised if they 
are violated.
"""
import sys
import inspect
import traceback
from functools import wraps
from types import GenericAlias, UnionType, TracebackType, FunctionType, MethodType



class ArgumentError(Exception):

	"""
	This class is used to construct the error message for restricted type violations and 
	to be caught in the methods ``@restrict`` wrapper call as distinct from other errors that 
	might have occured further down the traceback.
	"""
	
	def __init__(self, arg, arg_type, arg_name, name):
		"""
		The ArgumentError is used to construct the Error message for a restricted type violation.
		
		Parameters
		----------
		arg : object
			The argument given to the function.
		arg_type : type
			The type as restricted by the function signature.
		arg_name : str
			The name of the argument from the function signature.
		name : str
			The name of the method or function called.
		"""
		arg_str = str(arg)
		if arg_str.count('\n') > 8:
			arg_lines = arg_str.split()
			arg_str = '\n'.join(arg_lines[:4]) + '\n[…]\n' + '\n'.join(arg_lines[-4:])
		elif len(arg_str) > 1024:
			arg_str = arg_str[:100] + '\n[…]\n' + arg_str[-100:]
		super().__init__(f"""
A type restriction in function {name} was violated: 
{arg_name} is supposed to be of type {arg_type}, but got {type(arg)} instead.
The recieved argument for {arg_name} was:
{arg_str}
""")



class TypeChecker:

	"""
	TypeChecker controls function and class methods arguments for their validity.
	If the given type of an argument violates the type hint, a TypeError is raised.
	Use TypeChecker.restrict as a decorator to restrict a function or methods types. 
	"""
	
	@classmethod
	def restrict(cls, 
		     func: FunctionType|MethodType):
		"""
		Used as a decorator it ansurses that the decorated methods or functions
		arguments are type restricted according to the arguments type hints if provided.
		You can use None as an optional type hint instead of Nonetype.
		
		Parameters
		----------
		func : FunctionType | MethodType
			A function or method to be type restricted.
		
		Returns
		-------
		TYPE
			A wrapped function that resitrcts argument and return value types and raises a TypeError if violated.
		"""
		if not hasattr(func, '__code__') or not hasattr(func, '__annotations__'):
			return func
		code        = func.__code__
		annotations = func.__annotations__
		var_names   = code.co_varnames
		n_args      = code.co_argcount
		arg_names   = var_names[:n_args]
		arg_types   = [annotations[arg] for arg in arg_names if arg in annotations]
		static      = isinstance(func, (staticmethod, property, classmethod))
		if 'return' in annotations:
			return_type = annotations['return']
		else:
			return_type = False
		@wraps(func)
		def wrapper(*args, **kwargs):
			try:
				for arg, arg_type, arg_name in zip(args[0 if static else 1:], arg_types, arg_names[0 if static else 1:]):
					cls.__validate(arg=arg, 
						       arg_name=arg_name, 
						       arg_type=arg_type, 
						       name=func.__qualname__)
				for arg_name, arg in kwargs.items():
					if arg_name not in arg_names or arg_name not in annotations:
						continue
					cls.__validate(arg=arg, 
						       arg_name=arg_name, 
						       arg_type=annotations[arg_name], 
						       name=func.__qualname__)
				result = func(*args, **kwargs)
				if not return_type is False: # DO NOT REDUCE TO if not return_type!
					cls.__validate(arg=result, 
						       arg_name='the returned value', 
						       arg_type=return_type, 
						       name=func.__qualname__)
				return result
			except ArgumentError as error:
				raise TypeError(error) from None
			except Exception as error:
				tb  = error.__traceback__
				skip = lambda tb: tb.tb_frame.f_globals['__file__'] == __file__
				tbs = []
				while tb is not None:
					if not skip(tb):
						tbs.append(tb)
					tb = tb.tb_next
				if tbs and not skip(tbs[-1]):
					tbs[-1].tb_next = None
				for tb_A, tb_B in zip(tbs[:-1], tbs[1:]):
					tb_A.tb_next = tb_B
				if tbs:
					error.__traceback__ = tbs[0]
				raise error

		wrapper.__annotations__ = annotations
		if hasattr(func, __doc__):
			wrapper.__doc__ = func.__doc__
		return wrapper


	@classmethod
	def __validate(cls, 
		       arg:      object, 
		       arg_name: str, 
		       arg_type: type, 
		       name:     str):
		"""
		Helper method for type validation.
		
		Parameters
		----------
		arg : object
			An argument to bec checked.
		arg_name : str
			The name of he argument to be checked.
		arg_type : type
			The type convention the argument must abide by.
		name : str
			The name of the function or method that is validated.
		
		Raises
		------
		ArgumentError
		Description
		
		No Longer Raises
		----------------
		TypeError
		If the arg_types are violated a TypeError is raised.
		"""
		if not cls.__validate_type(arg, arg_type):
			raise ArgumentError(arg, arg_type, arg_name, name)


	@classmethod
	def __validate_type(cls, 
			    arg:      object, 
			    arg_type: type) -> bool:
		"""
		Helper method to check whether the arg_type is vioated.
		
		Parameters
		----------
		arg : object
			The argument to be checked.
		arg_type : type
			One or multiple argument types to be checked against.
		
		Returns
		-------
		bool
			Indicates wether the arg_type was valid according to arg_type.
		"""
		if isinstance(arg_type, GenericAlias):
			if not hasattr(arg, '__iter__'):
				return False
			seq_valid   = isinstance(arg, arg_type.__origin__)
			unroll = lambda x, types: any(cls.__validate_type(x, single) for single in types)
			items_valid = all(map(lambda x, types=arg_type.__args__: unroll(x, types), arg))
			return seq_valid and items_valid
		elif isinstance(arg_type, UnionType):
			return any(cls.__validate_type(arg, sub_arg_type) for sub_arg_type in arg_type.__args__)
		elif arg_type is None:
			return arg is None
		else:
			return isinstance(arg, arg_type)



restrict = TypeChecker.restrict



if __name__ == '__main__':
	@TypeChecker.restrict
	def f(x: (int, float), 
	      k: set[int] | dict, 
	      l: list[int | float], 
	      s: str='string', 
	      **kwargs) -> None | int:
		"""Summary
		
		Parameters
		----------
		x : int, float
			Description
		k : set[int] | dict
			Description
		l : list[int | float]
			Description
		s : str, optional
			Description
		**kwargs
			Description
		
		Returns
		-------
		None | int
		"""
		m = None
		print('hi')
		return 0
	def g():
		"""Summary
		"""
		f(3.2, k=0, l=[1, 2.3, 0])
	g()
	