import math
import unittest

import numpy as np

from gravity import (
    DEFAULT_MAX_DENOMINATOR,
    Rational,
    as_rational_array,
    rationalize,
    zeros,
    zeros_like,
)


class RationalTests(unittest.TestCase):
    def test_simplification_and_properties(self):
        value = Rational(10, 20)
        self.assertEqual(value.numerator, 1)
        self.assertEqual(value.denominator, 2)
        self.assertEqual(value.max_denominator, DEFAULT_MAX_DENOMINATOR)

    def test_arithmetic_operations(self):
        a = Rational(1, 3)
        b = Rational(1, 6)
        self.assertEqual(a + b, Rational(1, 2))
        self.assertEqual(a - b, Rational(1, 6))
        self.assertEqual(a * b, Rational(1, 18))
        self.assertEqual(a / b, Rational(2))

    def test_from_float(self):
        half = Rational.from_float(0.5)
        self.assertEqual(half, Rational(1, 2))

        pi_approx = Rational.from_float(math.pi, max_denominator=1000)
        self.assertLessEqual(pi_approx.denominator, 1000)
        self.assertAlmostEqual(float(pi_approx), math.pi, places=3)

    def test_rationalize_helper(self):
        value = rationalize(3.25, max_denominator=100)
        self.assertEqual(value, Rational(13, 4, max_denominator=100))

    def test_max_denominator_enforced_on_creation(self):
        value = Rational(123456, 987654)
        self.assertLessEqual(value.denominator, DEFAULT_MAX_DENOMINATOR)

    def test_constructor_accepts_real_components(self):
        dx = Rational(1, 3)
        value = Rational(1, 12 * float(dx))
        self.assertAlmostEqual(float(value), 1.0 / (12 * float(dx)))

    def test_list_broadcasting_and_numpy_interop(self):
        vector = [Rational(1, 2), Rational(2, 3)]
        shifted = Rational(1, 6) + vector
        self.assertTrue(all(isinstance(item, Rational) for item in shifted))
        np.testing.assert_allclose([float(item) for item in shifted], [2 / 3, 5 / 6])

        diff = vector - Rational(1, 3)
        self.assertTrue(all(isinstance(item, Rational) for item in diff))
        np.testing.assert_allclose([float(item) for item in diff], [1 / 6, 1 / 3])

        omega = Rational(3, 2)
        x0 = Rational(1, 5)
        profile = np.exp(-omega * (vector - x0) ** 2)
        self.assertIsInstance(profile, np.ndarray)
        np.testing.assert_allclose(
            [float(item) for item in profile],
            np.exp(-1.5 * (np.array([0.5, 2 / 3]) - 0.2) ** 2),
        )

    def test_numpy_array_operations_with_scalar(self):
        vector = np.array([0.25, 0.5, 0.75])
        result = Rational(1, 4) + vector
        self.assertEqual(result.dtype, object)
        self.assertTrue(all(isinstance(item, Rational) for item in result))
        np.testing.assert_allclose([float(item) for item in result], [0.5, 0.75, 1.0])

    def test_numpy_array_operations_with_object_array(self):
        vector = np.array([Rational(1, 2), Rational(1, 3)], dtype=object)
        result = vector + Rational(1, 6)
        np.testing.assert_allclose([float(item) for item in result], [2 / 3, 1 / 2])

    def test_numpy_ufunc_support(self):
        vector = np.array([Rational(1, 2), Rational(3, 4)], dtype=object)
        result = np.add(vector, Rational(1, 4))
        np.testing.assert_allclose([float(item) for item in result], [0.75, 1.0])

    def test_power_with_integer_exponent(self):
        value = Rational(2, 3)
        self.assertEqual(value ** 2, Rational(4, 9))
        self.assertEqual(value ** -1, Rational(3, 2))
        with self.assertRaises(ValueError):
            _ = value ** Rational(1, 2)

    def test_numpy_power(self):
        vector = np.array([Rational(2, 3), Rational(4, 5)], dtype=object)
        result = np.power(vector, 2)
        np.testing.assert_allclose([float(item) for item in result], [4 / 9, 16 / 25])

    def test_rational_array_helpers(self):
        arr = zeros(4)
        self.assertEqual(arr.shape, (4,))
        self.assertTrue(all(isinstance(item, Rational) for item in arr))

        base = [Rational(1, 2), 0.25, 0.75]
        arr_from_list = as_rational_array(base)
        self.assertEqual(arr_from_list.shape, (3,))
        self.assertTrue(all(isinstance(item, Rational) for item in arr_from_list))

        arr_like = zeros_like(arr_from_list)
        self.assertEqual(arr_like.shape, arr_from_list.shape)
        self.assertTrue(all(float(item) == 0.0 for item in arr_like))

    def test_formatting_support(self):
        value = Rational(3, 2)
        self.assertEqual(f"{value}", "3/2")
        self.assertEqual(f"{value:.2f}", "1.50")
        self.assertEqual(f"{value:.2e}", f"{float(value):.2e}")
        self.assertEqual(f"{value:r}", "3/2")


if __name__ == "__main__":  # pragma: no cover - direct execution helper
    unittest.main()
