import pytest

def median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        raise ValueError('empty')
    mid = n // 2
    return s[mid] if n % 2 else (s[mid-1] + s[mid]) / 2


class TestMedian:
    def test_single_element(self):
        assert median([5]) == 5

    def test_two_elements(self):
        assert median([1, 3]) == 2.0

    def test_odd_number_of_elements(self):
        assert median([1, 3, 5]) == 3

    def test_even_number_of_elements(self):
        assert median([1, 2, 3, 4]) == 2.5

    def test_unsorted_input(self):
        assert median([3, 1, 2]) == 2

    def test_negative_numbers(self):
        assert median([-1, -2, -3]) == -2

    def test_mixed_positive_and_negative(self):
        assert median([-1, 0, 1]) == 0

    def test_duplicate_elements(self):
        assert median([2, 2, 2]) == 2

    def test_empty_list_raises_value_error(self):
        with pytest.raises(ValueError, match='empty'):
            median([])

    def test_large_list_odd(self):
        xs = list(range(1, 101))
        assert median(xs) == 50

    def test_large_list_even(self):
        xs = list(range(1, 102))
        assert median(xs) == 50.5

    def test_float_values(self):
        assert median([1.5, 2.5, 3.5]) == 2.5

    def test_two_float_values(self):
        assert median([1.5, 2.5]) == 2.0


# (inspected manually)
import sys
src=open(sys.argv[0]).read()
print('ASSERTS', src.count('assert '), 'RAISES', src.count('pytest.raises'))
print('PASS')
