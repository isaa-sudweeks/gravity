"""Rational number utilities with NumPy interoperability."""
from __future__ import annotations

import math
import numbers
import operator
from fractions import Fraction
from typing import Any, Optional, Tuple, Union

try:  # NumPy is optional but recommended for array workflows.
    import numpy as np  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency may be absent.
    np = None  # type: ignore

NumberLike = Union["Rational", Fraction, numbers.Real]

DEFAULT_MAX_DENOMINATOR = 10**15


def _coerce_to_fraction(value: Any, *, max_denominator: int) -> Fraction:
    """Convert *value* into a :class:`Fraction` respecting ``max_denominator``."""

    if isinstance(value, Rational):
        return Fraction(value._numerator, value._denominator)
    if isinstance(value, Fraction):
        # Avoid needlessly truncating precise Fractions provided by the caller.
        return value
    if isinstance(value, numbers.Integral):
        return Fraction(int(value), 1)
    if np is not None and isinstance(value, np.generic):  # NumPy scalars
        return _coerce_to_fraction(value.item(), max_denominator=max_denominator)
    if isinstance(value, numbers.Real):
        frac = Fraction.from_float(float(value))
        if abs(frac.denominator) > max_denominator:
            frac = frac.limit_denominator(max_denominator)
        return frac
    raise TypeError(f"Cannot interpret {type(value)!r} as a rational component")


class Rational:
    """Representation of a rational number with automatic simplification."""

    __slots__ = ("_numerator", "_denominator", "_max_denominator")
    __array_priority__ = 1000.0  # Prefer Rational semantics in NumPy expressions.

    def __init__(
        self,
        numerator: NumberLike = 0,
        denominator: NumberLike = 1,
        *,
        max_denominator: Optional[int] = None,
    ) -> None:
        if max_denominator is None:
            max_denominator = DEFAULT_MAX_DENOMINATOR
        if max_denominator < 1:
            raise ValueError("max_denominator must be >= 1")

        num_fraction = _coerce_to_fraction(numerator, max_denominator=max_denominator)
        den_fraction = _coerce_to_fraction(denominator, max_denominator=max_denominator)
        if den_fraction == 0:
            raise ZeroDivisionError("denominator must be non-zero")

        combined = num_fraction / den_fraction
        num = combined.numerator
        den = combined.denominator

        num, den = self._normalize(num, den, max_denominator)

        self._numerator = num
        self._denominator = den
        self._max_denominator = max_denominator

    # ------------------------------------------------------------------
    # Constructors
    @classmethod
    def from_float(
        cls, value: float, *, max_denominator: Optional[int] = None
    ) -> "Rational":
        """Return the best rational approximation of *value*."""
        if isinstance(value, bool):  # bool is a subclass of int; treat explicitly.
            return cls(int(value), 1, max_denominator=max_denominator)
        if math.isnan(value) or math.isinf(value):
            raise ValueError("cannot convert NaN or infinity to Rational")
        if max_denominator is None:
            max_denominator = DEFAULT_MAX_DENOMINATOR
        frac = Fraction.from_float(value).limit_denominator(max_denominator)
        return cls(frac.numerator, frac.denominator, max_denominator=max_denominator)

    @classmethod
    def from_fraction(
        cls, value: Fraction, *, max_denominator: Optional[int] = None
    ) -> "Rational":
        """Create a :class:`Rational` from :class:`fractions.Fraction`."""
        if max_denominator is None:
            max_denominator = max(DEFAULT_MAX_DENOMINATOR, abs(value.denominator))
        frac = value.limit_denominator(max_denominator)
        return cls(frac.numerator, frac.denominator, max_denominator=max_denominator)

    @classmethod
    def rationalize(
        cls, value: NumberLike, *, max_denominator: Optional[int] = None
    ) -> "Rational":
        """Coerce a numeric-like value into :class:`Rational`."""
        if isinstance(value, Rational):
            if max_denominator is None or max_denominator == value._max_denominator:
                return value
            return value.limit_denominator(max_denominator)
        if isinstance(value, Fraction):
            return cls.from_fraction(value, max_denominator=max_denominator)
        if isinstance(value, numbers.Integral):
            return cls(int(value), 1, max_denominator=max_denominator)
        if np is not None and isinstance(value, np.generic):
            return cls.rationalize(value.item(), max_denominator=max_denominator)
        if isinstance(value, numbers.Real):
            return cls.from_float(float(value), max_denominator=max_denominator)
        raise TypeError(f"Cannot convert {type(value)!r} to Rational")

    # ------------------------------------------------------------------
    # Properties and helpers
    @property
    def numerator(self) -> int:
        return self._numerator

    @property
    def denominator(self) -> int:
        return self._denominator

    @property
    def max_denominator(self) -> int:
        return self._max_denominator

    def as_fraction(self) -> Fraction:
        """Return a :class:`Fraction` with the same value."""
        return Fraction(self._numerator, self._denominator)

    def limit_denominator(self, max_denominator: Optional[int] = None) -> "Rational":
        """Return the closest representable :class:`Rational` with limited denominator."""
        if max_denominator is None:
            max_denominator = self._max_denominator
        fraction = Fraction(self._numerator, self._denominator).limit_denominator(max_denominator)
        return Rational(
            fraction.numerator,
            fraction.denominator,
            max_denominator=max_denominator,
        )

    def exp(self, *, max_denominator: Optional[int] = None) -> "Rational":
        """Return ``e`` raised to this value as a :class:`Rational` approximation."""

        if max_denominator is None:
            max_denominator = self._max_denominator
        value = math.exp(float(self))
        return Rational.from_float(value, max_denominator=max_denominator)

    def sqrt(self, *, max_denominator: Optional[int] = None) -> "Rational":
        """Return the (principal) square root within the available precision."""

        if self._numerator < 0:
            raise ValueError("square root is undefined for negative Rational values")
        if max_denominator is None:
            max_denominator = self._max_denominator
        if self._numerator == 0:
            return Rational(0, 1, max_denominator=max_denominator)

        num_sqrt = math.isqrt(self._numerator)
        den_sqrt = math.isqrt(self._denominator)
        if num_sqrt * num_sqrt == self._numerator and den_sqrt * den_sqrt == self._denominator:
            # Preserve exact square roots when both components are perfect squares.
            return Rational(num_sqrt, den_sqrt, max_denominator=max_denominator)

        approx = math.sqrt(float(self))
        return Rational.from_float(approx, max_denominator=max_denominator)

    # ------------------------------------------------------------------
    # Numeric protocol
    def __float__(self) -> float:  # pragma: no cover - trivial mapping
        return self._numerator / self._denominator

    def __int__(self) -> int:  # pragma: no cover - trivial mapping
        return int(self._numerator // self._denominator)

    def __bool__(self) -> bool:  # pragma: no cover - trivial mapping
        return self._numerator != 0

    # ------------------------------------------------------------------
    # Representation
    def __repr__(self) -> str:
        return f"Rational({self._numerator}, {self._denominator})"

    def __str__(self) -> str:  # pragma: no cover - simple formatting
        if self._denominator == 1:
            return str(self._numerator)
        return f"{self._numerator}/{self._denominator}"

    def __format__(self, format_spec: str) -> str:  # pragma: no cover - simple delegation
        if format_spec in ("", "r", "R"):
            return str(self)
        try:
            return format(float(self), format_spec)
        except (ValueError, TypeError):
            return format(str(self), format_spec)

    # ------------------------------------------------------------------
    # Internal helpers
    def _coerce_scalar(self, value: Any) -> "Rational":
        if isinstance(value, Rational):
            return value
        if isinstance(value, Fraction):
            max_den = max(self._max_denominator, abs(value.denominator))
            return Rational.from_fraction(value, max_denominator=max_den)
        if isinstance(value, numbers.Integral):
            return Rational(int(value), 1, max_denominator=self._max_denominator)
        if np is not None and isinstance(value, np.generic):  # NumPy scalars
            return self._coerce_scalar(value.item())
        if isinstance(value, numbers.Real):
            return Rational.from_float(float(value), max_denominator=self._max_denominator)
        raise TypeError(f"Cannot interpret {type(value)!r} as Rational")

    def _vectorize_iterable(self, iterable, func):
        mapped = [func(item) for item in iterable]
        if np is not None:
            return np.array(mapped, dtype=object)
        if isinstance(iterable, tuple):
            return tuple(mapped)
        return mapped

    def _binary_operation(self, other: Any, op):
        if np is not None and isinstance(other, np.ndarray):
            vectorised = np.vectorize(
                lambda x: op(self, self._coerce_scalar(x)),
                otypes=[object],
            )
            return vectorised(other)
        if isinstance(other, (list, tuple)):
            return self._vectorize_iterable(
                other,
                lambda x: op(self, self._coerce_scalar(x)),
            )
        other_rat = self._coerce_scalar(other)
        return op(self, other_rat)

    @staticmethod
    def _combine_max_denominator(a: "Rational", b: "Rational") -> int:
        return max(a._max_denominator, b._max_denominator)

    @staticmethod
    def _normalize(num: int, den: int, max_denominator: int) -> Tuple[int, int]:
        if den < 0:
            num, den = -num, -den
        gcd = math.gcd(num, den)
        num //= gcd
        den //= gcd
        if abs(den) > max_denominator:
            fraction = Fraction(num, den).limit_denominator(max_denominator)
            num, den = fraction.numerator, fraction.denominator
        return num, den

    def _coerce_power(self, value: Any) -> int:
        if isinstance(value, numbers.Integral):
            return int(value)
        if isinstance(value, Rational):
            if value.denominator != 1:
                raise ValueError("Exponent must be an integer")
            return value.numerator
        if np is not None and isinstance(value, np.generic):
            return self._coerce_power(value.item())
        if isinstance(value, numbers.Real):
            if not float(value).is_integer():
                raise ValueError("Exponent must be an integer")
            return int(value)
        raise TypeError("Unsupported exponent type")

    # ------------------------------------------------------------------
    # Arithmetic operators
    def __add__(self, other: Any) -> Any:
        def _add(a: "Rational", b: "Rational") -> "Rational":
            max_den = self._combine_max_denominator(a, b)
            return Rational(
                a._numerator * b._denominator + b._numerator * a._denominator,
                a._denominator * b._denominator,
                max_denominator=max_den,
            )

        return self._binary_operation(other, _add)

    def __radd__(self, other: Any) -> Any:
        return self.__add__(other)

    def __sub__(self, other: Any) -> Any:
        def _sub(a: "Rational", b: "Rational") -> "Rational":
            max_den = self._combine_max_denominator(a, b)
            return Rational(
                a._numerator * b._denominator - b._numerator * a._denominator,
                a._denominator * b._denominator,
                max_denominator=max_den,
            )

        return self._binary_operation(other, _sub)

    def __rsub__(self, other: Any) -> Any:
        if np is not None and isinstance(other, np.ndarray):
            vectorised = np.vectorize(
                lambda x: self._coerce_scalar(x).__sub__(self),
                otypes=[object],
            )
            return vectorised(other)
        if isinstance(other, (list, tuple)):
            return self._vectorize_iterable(
                other,
                lambda x: self._coerce_scalar(x).__sub__(self),
            )
        return self._coerce_scalar(other).__sub__(self)

    def __mul__(self, other: Any) -> Any:
        def _mul(a: "Rational", b: "Rational") -> "Rational":
            max_den = self._combine_max_denominator(a, b)
            return Rational(
                a._numerator * b._numerator,
                a._denominator * b._denominator,
                max_denominator=max_den,
            )

        return self._binary_operation(other, _mul)

    def __rmul__(self, other: Any) -> Any:
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> Any:
        def _truediv(a: "Rational", b: "Rational") -> "Rational":
            if b._numerator == 0:
                raise ZeroDivisionError("division by zero")
            max_den = self._combine_max_denominator(a, b)
            return Rational(
                a._numerator * b._denominator,
                a._denominator * b._numerator,
                max_denominator=max_den,
            )

        return self._binary_operation(other, _truediv)

    def __rtruediv__(self, other: Any) -> Any:
        if np is not None and isinstance(other, np.ndarray):
            vectorised = np.vectorize(
                lambda x: self._coerce_scalar(x).__truediv__(self),
                otypes=[object],
            )
            return vectorised(other)
        if isinstance(other, (list, tuple)):
            return self._vectorize_iterable(
                other,
                lambda x: self._coerce_scalar(x).__truediv__(self),
            )
        return self._coerce_scalar(other).__truediv__(self)

    def __pow__(self, exponent: Any) -> Any:
        if np is not None and isinstance(exponent, np.ndarray):
            vectorised = np.vectorize(lambda x: self.__pow__(x), otypes=[object])
            return vectorised(exponent)
        power = self._coerce_power(exponent)
        if power >= 0:
            return Rational(
                self._numerator ** power,
                self._denominator ** power,
                max_denominator=self._max_denominator,
            )
        if self._numerator == 0:
            raise ZeroDivisionError("0 cannot be raised to a negative power")
        positive = -power
        return Rational(
            self._denominator ** positive,
            self._numerator ** positive,
            max_denominator=self._max_denominator,
        )

    def __neg__(self) -> "Rational":
        return Rational(
            -self._numerator,
            self._denominator,
            max_denominator=self._max_denominator,
        )

    def __pos__(self) -> "Rational":  # pragma: no cover - trivial
        return self

    def __abs__(self) -> "Rational":  # pragma: no cover - trivial
        return Rational(
            abs(self._numerator),
            self._denominator,
            max_denominator=self._max_denominator,
        )

    # ------------------------------------------------------------------
    # Comparisons
    def _compare(self, other: Any, op) -> bool:
        other_rat = self._coerce_scalar(other)
        return op(
            self._numerator * other_rat._denominator,
            other_rat._numerator * self._denominator,
        )

    def __eq__(self, other: Any) -> bool:
        try:
            return self._compare(other, operator.eq)
        except TypeError:
            return False

    def __lt__(self, other: Any) -> bool:
        return self._compare(other, operator.lt)

    def __le__(self, other: Any) -> bool:
        return self._compare(other, operator.le)

    def __gt__(self, other: Any) -> bool:  # pragma: no cover - mirrors __lt__
        return self._compare(other, operator.gt)

    def __ge__(self, other: Any) -> bool:  # pragma: no cover - mirrors __le__
        return self._compare(other, operator.ge)

    def __hash__(self) -> int:  # pragma: no cover - aligns with equality
        return hash((self._numerator, self._denominator))

    # ------------------------------------------------------------------
    # NumPy interoperability
    if np is not None:
        _UFUNC_DISPATCH = {
            np.add: operator.add,
            np.subtract: operator.sub,
            np.multiply: operator.mul,
            np.divide: operator.truediv,
            np.true_divide: operator.truediv,
            np.negative: operator.neg,
            np.positive: operator.pos,
            np.absolute: abs,
            np.power: operator.pow,
            np.exp: lambda a: a.exp(),
            np.sqrt: lambda a: a.sqrt(),
        }
    else:  # pragma: no cover - executed when NumPy unavailable
        _UFUNC_DISPATCH = {}

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):  # pragma: no cover - exercised indirectly
        if np is None:
            return NotImplemented
        if method != "__call__":
            return NotImplemented
        if kwargs.get("out") is not None:
            raise NotImplementedError("`out` argument is not supported for Rational ufuncs")
        op = self._UFUNC_DISPATCH.get(ufunc)
        if op is None:
            return NotImplemented

        coerced = []
        has_array = False
        for value in inputs:
            if isinstance(value, Rational):
                coerced.append(value)
            elif isinstance(value, np.ndarray):
                vectorised = np.vectorize(lambda x: self._coerce_scalar(x), otypes=[object])
                coerced.append(vectorised(value))
                has_array = True
            else:
                coerced.append(self._coerce_scalar(value))
        if has_array:
            vectorised = np.vectorize(lambda *args: op(*args), otypes=[object])
            return vectorised(*coerced)
        return op(*coerced)


def rationalize(value: NumberLike, *, max_denominator: Optional[int] = None) -> Rational:
    """Public helper to convert *value* into :class:`Rational`."""

    return Rational.rationalize(value, max_denominator=max_denominator)


def exp(value: NumberLike, *, max_denominator: Optional[int] = None) -> Rational:
    """Return ``e`` raised to *value* as a :class:`Rational` approximation."""

    rational_value = Rational.rationalize(value, max_denominator=max_denominator)
    return rational_value.exp(max_denominator=max_denominator)


def as_rational_array(
    values: Any,
    *,
    max_denominator: Optional[int] = None,
    copy: bool = True,
) -> "np.ndarray":
    """Return a ``numpy.ndarray`` of :class:`Rational` values.

    ``values`` can be any iterable containing numeric-like entries or an existing
    NumPy array. When ``copy`` is ``False`` and ``values`` is already a NumPy
    array with ``dtype=object``, the original array is returned after ensuring
    each element is a :class:`Rational`.
    """

    if np is None:
        raise RuntimeError("NumPy is required to construct Rational arrays")

    if isinstance(values, np.ndarray):
        if copy:
            array = values.copy()
        else:
            array = values
        if array.dtype != object:
            array = array.astype(object, copy=False)
        if array.dtype == object and all(isinstance(item, Rational) for item in array.flat):
            return array
        vectorised = np.vectorize(
            lambda item: Rational.rationalize(item, max_denominator=max_denominator),
            otypes=[object],
        )
        return vectorised(array)

    if isinstance(values, (list, tuple)):
        coerced = [
            Rational.rationalize(item, max_denominator=max_denominator) for item in values
        ]
        return np.array(coerced, dtype=object)

    return as_rational_array(list(values), max_denominator=max_denominator, copy=copy)


def zeros(
    length: int,
    *,
    max_denominator: Optional[int] = None,
) -> "np.ndarray":
    """Return a one-dimensional array of length ``length`` filled with zeros."""

    if length < 0:
        raise ValueError("length must be non-negative")
    return as_rational_array(
        [Rational(0, 1, max_denominator=max_denominator) for _ in range(length)],
        copy=True,
    )


def zeros_like(
    values: Any,
    *,
    max_denominator: Optional[int] = None,
    copy: bool = True,
) -> "np.ndarray":
    """Return a zero-filled array that matches the shape of ``values``."""

    array = as_rational_array(values, max_denominator=max_denominator, copy=copy)
    zeros_flat = [
        Rational(0, 1, max_denominator=max_denominator) for _ in range(array.size)
    ]
    return np.array(zeros_flat, dtype=object).reshape(array.shape)


__all__ = [
    "Rational",
    "rationalize",
    "DEFAULT_MAX_DENOMINATOR",
    "exp",
    "as_rational_array",
    "zeros",
    "zeros_like",
]
