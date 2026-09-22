import re
import numpy as np
import pandas as pd
from bokehlab import parse, plot

def compare(a, b):
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        for q, r in zip(a, b):
            if not compare(q, r):
                return False
    elif isinstance(a, (np.ndarray, pd.Index, pd.Series)) or \
         isinstance(b, (np.ndarray, pd.Index, pd.Series)):
        a, b = np.array(a), np.array(b)
        if a.shape != b.shape or not np.allclose(a, b):
            return False
    elif a != b:
        return False
    return True

def assert_equal(a, b):
    assert compare(a, b), f"{a!r} != {b!r}"

def check_parse_arr(x, x1, y, y1):
    x0 = [0, 1, 2]
    assert_equal(parse(y), [(x0, y, '-', 'a', None)])

    assert_equal(parse(y), [(x0, y, '-', 'a', None)])
    assert_equal(parse(y, '.-'), [(x0, y, '.-', 'a', None)])
    assert_equal(parse(y, style='.-'), [(x0, y, '.-', 'a', None)])
    assert_equal(parse(y, color='g'), [(x0, y, '-', 'g', None)])
    assert_equal(parse(y, '.-', 'g'), [(x0, y, '.-', 'g', None)])
    assert_equal(parse(y, '.-', 'g', label='y'), [(x0, y, '.-', 'g', 'y')])

    assert_equal(parse(x, y), [(x, y, '-', 'a', None)])
    assert_equal(parse(x, y, '.-'), [(x, y, '.-', 'a', None)])
    assert_equal(parse(x, y, '.-', 'g'), [(x, y, '.-', 'g', None)])
    assert_equal(parse(x, y, '.-', 'g', label='y'), [(x, y, '.-', 'g', 'y')])

    assert_equal(parse(x, y, style='.-'), [(x, y, '.-', 'a', None)])
    assert_equal(parse(x, y, color='g'), [(x, y, '-', 'g', None)])
    assert_equal(parse(x, y, label='y'), [(x, y, '-', 'a', 'y')])
    assert_equal(parse(x, y, style='.-', color='g'), [(x, y, '.-', 'g', None)])
    assert_equal(parse(x, y, style='.-', color='g', label='y'), [(x, y, '.-', 'g', 'y')])

    assert_equal(parse(x, [y, y1]), [(x, y, '-', 'a', None), (x, y1, '-', 'a', None)])
    assert_equal(parse(x, [y, y1], '.-'), [(x, y, '.-', 'a', None), (x, y1, '.-', 'a', None)])
    assert_equal(parse(x, [y, y1], ['.', '-']), [(x, y, '.', 'a', None), (x, y1, '-', 'a', None)])
    assert_equal(parse(x, [y, y1], '.-', ['r', 'g']), [(x, y, '.-', 'r', None), (x, y1, '.-', 'g', None)])
    assert_equal(parse(x, [y, y1], ['.', '-'], ['r', 'g']), [(x, y, '.', 'r', None), (x, y1, '-', 'g', None)])
    assert_equal(parse(x, [y, y1], ['.', '-'], ['r', 'g'], label=['y1', 'y2']), [(x, y, '.', 'r', 'y1'), (x, y1, '-', 'g', 'y2')])

    assert_equal(parse([x, x1], [y, y1]), [(x, y, '-', 'a', None), (x1, y1, '-', 'a', None)])
    assert_equal(parse([x, x1], [y, y1], '.-'), [(x, y, '.-', 'a', None), (x1, y1, '.-', 'a', None)])
    assert_equal(parse([x, x1], [y, y1], '.-', ['r', 'g']), [(x, y, '.-', 'r', None), (x1, y1, '.-', 'g', None)])
    assert_equal(parse([x, x1], [y, y1], ['.', '-'], ['b']), [(x, y, '.', 'b', None), (x1, y1, '-', 'b', None)])
    assert_equal(parse([x, x1], [y, y1], ['.', '-'], ['r', 'g']), [(x, y, '.', 'r', None), (x1, y1, '-', 'g', None)])
    assert_equal(parse([x, x1], [y, y1], ['.', '-'], ['r', 'g'], label=['y1', 'y2']), [(x, y, '.', 'r', 'y1'), (x1, y1, '-', 'g', 'y2')])

    assert_equal(parse([x, x1], y), [(x, y, '-', 'a', None), (x1, y, '-', 'a', None)])
    assert_equal(parse([x, x1], y, '.-'), [(x, y, '.-', 'a', None), (x1, y, '.-', 'a', None)])
    assert_equal(parse([x, x1], y, '.-', ['r', 'g']), [(x, y, '.-', 'r', None), (x1, y, '.-', 'g', None)])
    assert_equal(parse([x, x1], y, ['.', '-'], ['b']), [(x, y, '.', 'b', None), (x1, y, '-', 'b', None)])
    assert_equal(parse([x, x1], y, ['.', '-'], ['r', 'g']), [(x, y, '.', 'r', None), (x1, y, '-', 'g', None)])
    assert_equal(parse([x, x1], y, ['.', '-'], ['r', 'g'], label=['y1', 'y2']), [(x, y, '.', 'r', 'y1'), (x1, y, '-', 'g', 'y2')])
    print()

def test_parse_arrays():
    x = [1, 2, 3]
    x1 = [-1, -2, -3]
    y = [1, 4, 9]
    y1 = [-1, -4, -9]
    check_parse_arr(x, x1, y, y1)
    check_parse_arr(x, x1, np.array(y), y1)
    check_parse_arr(np.array(x), x1, y, y1)
    check_parse_arr(np.array(x), x1, np.array(y), y1)
    check_parse_arr(np.array(x), np.array(x1), np.array(y), np.array(y1))

def test_parse_np():
    x0 = [0, 1, 2]
    x = [1, 2, 3]
    x1 = [-1, -2, -3]
    y = [1, 4, 9]
    y1 = [-1, -4, -9]
    xx = np.array([[1, 2, 3], [-1, -2, -3]]).T
    yy = np.array([[1, 4, 9], [-1, -4, -9]]).T
    assert_equal(parse(yy), [(x0, y, '-', 'a', None), (x0, y1, '-', 'a', None)])
    assert_equal(parse(x, yy), [(x, y, '-', 'a', None), (x, y1, '-', 'a', None)]),
    assert_equal(parse(np.array(x), yy), [(x, y, '-', 'a', None), (x, y1, '-', 'a', None)]),
    assert_equal(parse(xx, y), [(x, y, '-', 'a', None), (x1, y, '-', 'a', None)]),
    assert_equal(parse(xx, yy), [(x, y, '-', 'a', None), (x1, y1, '-', 'a', None)])
    assert_equal(parse(xx, yy, '.'), [(x, y, '.', 'a', None), (x1, y1, '.', 'a', None)])
    assert_equal(parse(xx, yy, '.-', 'g'), [(x, y, '.-', 'g', None), (x1, y1, '.-', 'g', None)])
    print()

def test_parse_pd():
    x0 = np.array([0, 1, 2])
    x = np.array([1, 2, 3])
    x1 = np.array([-1, -2, -3])
    x2 = np.array([-10, -20, -30])
    y = np.array([1, 4, 9])
    y1 = np.array([-1, -4, -9])
    df = pd.DataFrame({'y': y, 'y1': y1})
    df1 = pd.DataFrame({'y': y, 'y1': y1}, index=x)
    assert_equal(parse(df), [(x0, y, '-', 'a', 'y'), (x0, y1, '-', 'a', 'y1')])
    assert_equal(parse(df1), [(x, y, '-', 'a', 'y'), (x, y1, '-', 'a', 'y1')])
    assert_equal(parse(x1, df), [(x1, y, '-', 'a', 'y'), (x1, y1, '-', 'a', 'y1')])
    assert_equal(parse(x1, df1), [(x1, y, '-', 'a', 'y'), (x1, y1, '-', 'a', 'y1')])
    assert_equal(parse([x1, x2], df), [(x1, y, '-', 'a', 'y'), (x2, y1, '-', 'a', 'y1')])
    assert_equal(parse([x1, x2], df1), [(x1, y, '-', 'a', 'y'), (x2, y1, '-', 'a', 'y1')])
    assert_equal(parse(df, label=['Y', 'Y1']), [(x0, y, '-', 'a', 'Y'), (x0, y1, '-', 'a', 'Y1')])
    print()

def test_parse_dicts():
    x0 = np.array([0, 1, 2])
    x = np.array([1, 2, 3])
    x1 = np.array([-1, -2, -3])
    y = np.array([1, 4, 9])
    y1 = np.array([-1, -4, -9])
    d = {'y': y, 'y1': y1}
    assert_equal(parse(d), [(x0, y, '-', 'a', 'y'), (x0, y1, '-', 'a', 'y1')])
    assert_equal(parse(x, d), [(x, y, '-', 'a', 'y'), (x, y1, '-', 'a', 'y1')])
    assert_equal(parse([x, x1], d), [(x, y, '-', 'a', 'y'), (x1, y1, '-', 'a', 'y1')])
    print()

def test_exceptions_1():
#    FIGURE.clear()
    import pytest
    with pytest.raises(ValueError):
        plot([[1,2,3],[4,5,6]], [[1,2,3],[4,5,6],[7,8,9]])
    with pytest.raises(ValueError, match='length of label = 1 must match the number of plots = 2'):
        plot([[1,2,3],[4,5,6]], label=['y'])
    with pytest.raises(ValueError, match=re.escape('len(alpha)=2 does not match len(y)=1')):
        plot([1,2,3], [-1,-2,-3], alpha=[0.1, 0.7])

