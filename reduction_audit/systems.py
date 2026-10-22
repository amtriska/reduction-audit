import itertools
import random
from dataclasses import dataclass

from sympy import GF, QQ
from sympy.polys.orderings import grevlex
from sympy.polys.rings import ring


@dataclass
class System:
    name: str
    ring: object
    polys: list


def _ring(n, domain):
    names = ["x", "y", "z", "w", "u", "v"][:n] if n <= 6 else [f"x{i}" for i in range(n)]
    return ring(",".join(names), domain, grevlex)[0]


def _poly(R, pairs):
    d = {}
    for mon, c in pairs:
        d[mon] = d.get(mon, 0) + c
    return R.from_dict({m: c for m, c in d.items() if c})


def _unit(n, i):
    return tuple(1 if j == i else 0 for j in range(n))


def _mul(a, b):
    return tuple(x + y for x, y in zip(a, b))


def cyclic(n):
    R = _ring(n, QQ)
    polys = []
    for d in range(1, n):
        pairs = []
        for i in range(n):
            mon = [0] * n
            for k in range(d):
                mon[(i + k) % n] += 1
            pairs.append((tuple(mon), 1))
        polys.append(_poly(R, pairs))
    polys.append(_poly(R, [((1,) * n, 1), ((0,) * n, -1)]))
    return System(f"cyclic-{n}", R, polys)


def katsura(n):
    nv = n + 1
    R = _ring(nv, QQ)
    u = [_unit(nv, i) for i in range(nv)]
    polys = [_poly(R, [(u[0], 1)] + [(u[i], 2) for i in range(1, nv)] + [((0,) * nv, -1)])]
    for k in range(n):
        pairs = [(_mul(u[abs(i)], u[abs(k - i)]), 1) for i in range(-n, n + 1) if abs(k - i) <= n]
        pairs.append((u[k], -1))
        polys.append(_poly(R, pairs))
    return System(f"katsura-{n}", R, polys)


def trap():
    R = _ring(3, QQ)
    return System("trap", R, [_poly(R, [((2, 0, 0), 1), ((0, 1, 0), -1)]),
                              _poly(R, [((1, 1, 0), 1), ((0, 0, 1), -1)])])


def random_quadratic(n, m, seed, p=32003):
    R = _ring(n, GF(p))
    rng = random.Random(seed)
    quad = [tuple(c.count(i) for i in range(n))
            for c in itertools.combinations_with_replacement(range(n), 2)]
    polys = [_poly(R, [(mon, rng.randrange(1, p)) for mon in quad]) for _ in range(m)]
    return System(f"mq-{n}-{m}-s{seed}-p{p}", R, polys)


def minors_2x2(matrix, nvars, name):
    R = _ring(nvars, QQ)
    u = [_unit(nvars, i) for i in range(nvars)]
    polys = []
    # the generator order matters under POT: (row pair, column pair), lexicographic
    for r1, r2 in itertools.combinations(range(len(matrix)), 2):
        for c1, c2 in itertools.combinations(range(len(matrix[0])), 2):
            polys.append(_poly(R, [(_mul(u[matrix[r1][c1]], u[matrix[r2][c2]]), 1),
                                   (_mul(u[matrix[r1][c2]], u[matrix[r2][c1]]), -1)]))
    return System(name, R, polys)


def twisted_cubic():
    return minors_2x2([[0, 1, 2], [1, 2, 3]], 4, "twisted-cubic")
