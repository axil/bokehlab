import re
from collections.abc import Iterable

import numpy as np
import pandas as pd

USE_TORCH = 0

if USE_TORCH:
    import torch
else:
    class torch:
        class Tensor:
            pass

def is_2d(y):
    if hasattr(y, 'shape'):      # numpy and torch
        return len(y.shape) > 1
#    if isinstance(y, torch.Tensor) and y.dim()==2:
#        return True
#    if not isinstance(y, torch.Tensor) and 
    return isinstance(y, Iterable) and len(y) and isinstance(y[0], Iterable)  # lists and tuples

class ParseError(Exception):
    pass

class Missing:
    pass

def parse_spec(spec):
    if spec is None:
        spec = ''
    style = re.sub('[a-z]', '', spec) or '-'
    color = re.sub('[^a-z]', '', spec) or 'a'
    return style, color

def parse3(x, y, spec): # -> list of (x, y, spec)
    tr = []
    #import ipdb; ipdb.set_trace()
    if is_2d(y):
        labels = None
        if isinstance(y, np.ndarray):
            yy = y.T
        elif isinstance(y, pd.DataFrame):
            labels = list(map(str, y.columns))
            yy = [y[col].values for col in labels]
        else:
            yy = y
        n = len(yy)    # number of plots
        if labels is None:
            labels = [None] * n
#        if isinstance(x, Missing) and w == 2 and h > 2:
#            if isinstance(y, np.ndarray):
#                tr.append((y[:,0], y[:,1], spec))
#            else:
#                _x, _y = [], []
#                for row in y:
#                    _x.append(row[0])
#                    _y.append(row[1])
#                tr.append((_x, _y, spec))
#        else:
        #n = len(y)
        if isinstance(spec, (tuple, list)):
            specs = spec
            if len(specs) != w:
                raise ParseError(f'len(spec)={len(spec)} does not match len(y)={len(y)}')
        else:
            style, colors = parse_spec(spec)
            if len(colors) == n:
                specs = [style+c for c in colors]
            else:
                specs = [style+colors]*n
        if is_2d(x):
            if hasattr(x, 'T'):
                x = x.T
            for xi, yi, si in zip(x, yy, specs):
                tr.append((xi, yi, si))
        else:
            for yi, si, lb in zip(yy, specs, labels):
                if isinstance(x, Missing):
                    xi = np.arange(len(yi))
                else:
                    xi = x
                tr.append((xi, yi, si, lb))
    else:
        if isinstance(x, Missing):
            x = np.arange(len(y))
        if isinstance(y, pd.DataFrame) and len(y.columns) > 0:
            label = y.columns[0]
        else:
            label = None
        tr.append((x, y, spec, label))
    return tr

def test_parse3():
    x = [1,2,3]
    y = [1,4,9]
    y1 = [-1,-4,-9]
    assert parse3(x, y, '') == [(x, y, '', None)]
    assert parse3(x, [y, y1], '') == [(x, y, '-a', None), (x, y1, '-a', None)]
    assert parse3(x, [y, y1], '.') == [(x, y, '.a', None), (x, y1, '.a', None)]
    assert parse3(x, [y, y1], '.-') == [(x, y, '.-a', None), (x, y1, '.-a', None)]
    assert parse3(x, [y, y1], 'gr') == [(x, y, '-g', None), (x, y1, '-r', None)]
    assert parse3(x, [y, y1], '.-gr') == [(x, y, '.-g', None), (x, y1, '.-r', None)]
    print(parse3(x, [y, y1], '.-gr'))

def parse(*args, color=None, label=None):
    tr = []
    style = '-'
    if len(args) in (1, 2):
        x = Missing()
        if len(args) == 1:
            y = args[0]
        elif isinstance(args[1], str) or \
             isinstance(args[1], (tuple, list)) and len(args[0]) > 0 and \
             isinstance(args[1][0], str):
            y, style = args
        else:
            x, y = args
        if isinstance(y, dict):
            x, y = list(y.keys()), list(y.values())
        tr.extend(parse3(x, y, style))
    elif len(args) % 3 == 0:
        n = len(args)//3
        for h in range(n):
            x, y, style = args[3*h:3*(h+1)]
            tr.extend(parse3(x, y, style))
    n = len(tr)

    # color
    if isinstance(color, (list, tuple)):
        if len(color) != n:
            raise ValueError(f'len(color)={len(color)}; color should either be a string or a tuple of length {n}')
        colors = color
    elif isinstance(color, str):
        colors = [color] * n
    elif color is None:
        colors = 'a'*n
    else:
        raise ValueError(f'color={color}; it should either be a string or a tuple of length {n}')

    # legend
    if isinstance(label, (list, tuple)):
        if len(label) != n:
            raise ValueError(f'len(label)={len(label)}; label should either be a string or a tuple of length {n}')
        labels = label
    elif isinstance(label, str):
        labels = [label] * n
    elif label is None:
        labels = [None] * n
    else:
        raise ValueError(f'label={label}; it should either be a string or a tuple of length {n}')
    qu = []
    try:
        for (x, y, spec, label1), color, label2 in zip(tr, colors, labels):
            style, _color = parse_spec(spec)
            if _color != 'a':
                color = _color
            qu.append((x, y, style, color, label2 if label2 is not None else label1))
    except Exception as e:
        print(e)
        
    return qu
    

def eq(a, b):
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        for q, r in zip(a, b):
            if isinstance(q, np.ndarray) or isinstance(r, np.ndarray):
                if not np.allclose(q, r):
                    return False
            else:
                if eq(q, r):
                    return False
        return True
    return a == b

def fast_tests():
    x = [1, 2, 3]
    y = [1, 4, 9]
    y1 = [-1, -4, -9]
    #assert parse3(x, y, '') == [(x, y, '')]
    #assert parse3(x, [y, y1], '') == [(x, y, '-a'), (x, y1, '-a')]
    #assert parse3(x, [y, y1], '.') == [(x, y, '.a'), (x, y1, '.a')]
    #assert parse3(x, [y, y1], '.-') == [(x, y, '.-a'), (x, y1, '.-a')]
    #assert parse3(x, [y, y1], 'gr') == [(x, y, '-g'), (x, y1, '-r')]
    #assert parse3(x, [y, y1], '.-gr') == [(x, y, '.-g'), (x, y1, '.-r')]
    assert eq(parse(y), [([0, 1, 2], y, '-', 'a', None)])
    assert parse(x, y) == [(x, y, '-', 'a', None)]
    assert parse(x, y, '.') == [(x, y, '.', 'a', None)]
    assert parse(x, y, '.-') == [(x, y, '.-', 'a', None)]
    assert parse(x, y, '.-g') == [(x, y, '.-', 'g', None)]
    assert parse(x, y, '.-g', label='aaa') == [(x, y, '.-', 'g', 'aaa')]
    assert parse(x, [y, y1], '.-', color=['r', 'g']) == [(x, y, '.-', 'r', None), (x, y1, '.-', 'g', None)]
    assert parse(x, [y, y1], '.-rg', label=['y', 'y1']) == [(x, y, '.-', 'r', 'y'), (x, y1, '.-', 'g', 'y1')]
    import pandas as pd
#    assert parse(x, [y, y1], '.-rg', label=['y', 'y1']) == [(x, y, '.-', 'r', 'y'), (x, y1, '.-', 'g', 'y1')]

    print(parse(x, pd.DataFrame({'y': y, 'y1': y1}), '.-g'))

if __name__ == '__main__':
    fast_tests()
#    import pytest
#    pytest.main(['-s', __file__+'::test_parser'])

    
