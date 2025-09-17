import math
import unittest

import numpy as np

from gravity import DEFAULT_MAX_DENOMINATOR, Rational, rationalize


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


if __name__ == "__main__":  # pragma: no cover - direct execution helper
    unittest.main()
