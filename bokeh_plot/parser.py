import re
from collections.abc import Iterable

USE_TORCH = 0

if USE_TORCH:
    import torch
else:
    class torch:
        class Tensor:
            pass

def is_2d(y):
    return isinstance(y, torch.Tensor) and y.dim()==2 or \
       not isinstance(y, torch.Tensor) and isinstance(y, Iterable) and len(y) and isinstance(y[0], Iterable)

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
        h, w = len(y), len(y[0])
        if isinstance(x, Missing) and w == 2 and h > 2:
            if isinstance(y, np.ndarray):
                tr.append((y[:,0], y[:,1], spec))
            else:
                _x, _y = [], []
                for row in y:
                    _x.append(row[0])
                    _y.append(row[1])
                tr.append((_x, _y, spec))
        else:
            n = len(y)
            if isinstance(spec, (tuple, list)):
                specs = spec
                if len(specs) != n:
                    raise ParseError(f'len(spec)={len(spec)} does not match len(y)={len(y)}')
            else:
                style, colors = parse_spec(spec)
                if len(colors) == n:
                    specs = [style+c for c in colors]
                else:
                    specs = [style+colors]*n
            if is_2d(x):
                for xi, yi, si in zip(x, y, specs):
                    tr.append((xi, yi, si))
            else:
                for yi, si in zip(y, specs):
                    if isinstance(x, Missing):
                        xi = list(range(len(yi)))
                    else:
                        xi = x
                    tr.append((xi, yi, si))
    else:
        if isinstance(x, Missing):
            x = list(range(len(y)))
        tr.append((x, y, spec))
    return tr

def test_parse3():
    x = [1,2,3]
    y = [1,4,9]
    y1 = [-1,-4,-9]
    assert parse3(x, y, '') == [(x, y, '')]
    assert parse3(x, [y, y1], '') == [(x, y, '-a'), (x, y1, '-a')]
    assert parse3(x, [y, y1], '.') == [(x, y, '.a'), (x, y1, '.a')]
    assert parse3(x, [y, y1], '.-') == [(x, y, '.-a'), (x, y1, '.-a')]
    assert parse3(x, [y, y1], 'gr') == [(x, y, '-g'), (x, y1, '-r')]
    assert parse3(x, [y, y1], '.-gr') == [(x, y, '.-g'), (x, y1, '.-r')]
    print(parse3(x, [y, y1], '.-gr'))

def parse(*args, color=None, legend=None):
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
    if isinstance(legend, (list, tuple)):
        if len(legend) != n:
            raise ValueError(f'len(legend)={len(legend)}; legend should either be a string or a tuple of length {n}')
        legends = legend
    elif isinstance(legend, str):
        legends = [legend] * n
    elif legend is None:
        legends = [None] * n
    else:
        raise ValueError(f'legend={legend}; it should either be a string or a tuple of length {n}')
    qu = []
    for (x, y, spec), color, legend in zip(tr, colors, legends):
        style, _color = parse_spec(spec)
        if _color != 'a':
            color = _color
        qu.append((x, y, style, color, legend))
    return qu
    
def test_parser():
    x = [1,2,3]
    y = [1,4,9]
    y1 = [-1,-4,-9]
    #assert parse3(x, y, '') == [(x, y, '')]
    #assert parse3(x, [y, y1], '') == [(x, y, '-a'), (x, y1, '-a')]
    #assert parse3(x, [y, y1], '.') == [(x, y, '.a'), (x, y1, '.a')]
    #assert parse3(x, [y, y1], '.-') == [(x, y, '.-a'), (x, y1, '.-a')]
    #assert parse3(x, [y, y1], 'gr') == [(x, y, '-g'), (x, y1, '-r')]
    #assert parse3(x, [y, y1], '.-gr') == [(x, y, '.-g'), (x, y1, '.-r')]
    assert parse(y) == [([0, 1, 2], y, '-', 'a', None)]
    assert parse(x, y) == [(x, y, '-', 'a', None)]
    assert parse(x, y, '.') == [(x, y, '.', 'a', None)]
    assert parse(x, y, '.-') == [(x, y, '.-', 'a', None)]
    assert parse(x, y, '.-g') == [(x, y, '.-', 'g', None)]
    assert parse(x, y, '.-g', legend='aaa') == [(x, y, '.-', 'g', 'aaa')]
    assert parse(x, [y, y1], '.-', color=['r', 'g']) == [(x, y, '.-', 'r', None), (x, y1, '.-', 'g', None)]
    assert parse(x, [y, y1], '.-rg', legend=['y', 'y1']) == [(x, y, '.-', 'r', 'y'), (x, y1, '.-', 'g', 'y1')]
    print(parse(x, [y, y1], '.-g', legend='aaa'))
