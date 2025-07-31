import sympy as sp
import functools
import torch

from typing import Any, Callable, TypeVar, Union, Dict, Tuple, List


T = TypeVar("T")


# Helper function to reduce multiple arguments using a binary function
def _reduce(fn: Callable[..., T]) -> Callable[..., T]:
    """
    Creates a reduction function that applies a binary operation repeatedly.
    Useful for converting n-ary Sympy operations to binary PyTorch operations.
    """

    def fn_(*args: Any) -> T:
        return functools.reduce(fn, args)

    return fn_


def _imaginary(*_: Any) -> torch.Tensor:
    """Returns the imaginary unit as a PyTorch tensor"""
    return torch.tensor(1j)


def make_tuple(value):
    if isinstance(value, int):
        return value
    elif isinstance(value, list):
        return tuple(value)
    return value


def make_2d_tuple(value):
    if isinstance(value, int):
        return (value, value)
    elif isinstance(value, list):
        return tuple(value)
    return value


def make_3d_tuple(value):
    if isinstance(value, int):
        return (value, value, value)
    elif isinstance(value, list):
        return tuple(value)
    return value


def single_to_int(value: Union[int, Tuple[int], List[int]]):
    if isinstance(value, int):
        return value
    assert len(value) == 1, "_single_to_int only for single values"
    return value[0]


# Mapping between Sympy operations and their PyTorch equivalents
SYMPY_TO_TORCH_OPS = {
    # Basic arithmetic
    sp.Mul: _reduce(torch.mul),
    sp.Add: _reduce(torch.add),
    sp.div: torch.div,
    sp.Pow: torch.pow,
    # Basic mathematical functions
    sp.Abs: torch.abs,
    sp.sign: torch.sign,
    sp.ceiling: torch.ceil,
    sp.floor: torch.floor,
    sp.log: torch.log,
    sp.exp: torch.exp,
    sp.sqrt: torch.sqrt,
    # Trigonometric functions
    sp.cos: torch.cos,
    sp.sin: torch.sin,
    sp.tan: torch.tan,
    sp.acos: torch.acos,
    sp.asin: torch.asin,
    sp.atan: torch.atan,
    sp.atan2: torch.atan2,
    # Hyperbolic functions
    sp.cosh: torch.cosh,
    sp.sinh: torch.sinh,
    sp.tanh: torch.tanh,
    sp.acosh: torch.acosh,
    sp.asinh: torch.asinh,
    sp.atanh: torch.atanh,
    # Complex operations
    sp.re: torch.real,
    sp.im: torch.imag,
    sp.arg: torch.angle,
    sp.core.numbers.ImaginaryUnit: _imaginary,
    sp.conjugate: torch.conj,
    # Special functions
    sp.erf: torch.erf,
    sp.loggamma: torch.lgamma,
    # Comparison operations
    sp.Eq: torch.eq,
    sp.Ne: torch.ne,
    sp.StrictGreaterThan: torch.gt,
    sp.StrictLessThan: torch.lt,
    sp.LessThan: torch.le,
    sp.GreaterThan: torch.ge,
    # Logical operations
    sp.And: torch.logical_and,
    sp.Or: torch.logical_or,
    sp.Not: torch.logical_not,
    # Min/Max operations
    sp.Max: torch.max,
    sp.Min: torch.min,
    # Matrix operations
    sp.MatAdd: torch.add,
    sp.HadamardProduct: torch.mul,
    sp.Trace: torch.trace,
    sp.Determinant: torch.det,
}
