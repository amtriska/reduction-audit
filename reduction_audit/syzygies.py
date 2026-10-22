from dataclasses import dataclass, field

from sympy import Symbol
from sympy.polys.groebnertools import groebner
from sympy.polys.monomials import monomial_divides
from sympy.polys.orderings import ProductOrder, grevlex, lex
from sympy.polys.rings import xring


def minimalise(monomials):
    out = []
    for g in sorted(set(monomials), key=sum):
        if not any(monomial_divides(o, g) for o in out):
            out.append(g)
    return out


def colon_lt(J, f, R):
    """Sorted minimal generators of LT(J : f) and of LT(J).

    J : f = (J ∩ <f>) / f, and J ∩ <f> is the t-free part of a Gröbner basis of
    tJ + (1 - t)f under a block order with t first.
    """
    if not J:
        return [], []
    tname = "t"
    while Symbol(tname) in R.symbols:
        tname = "_" + tname
    order = ProductOrder((lex, lambda m: m[:1]), (grevlex, lambda m: m[1:]))
    Rt, gens = xring((Symbol(tname),) + R.symbols, R.domain, order)
    t = gens[0]
    elim = [t * g.set_ring(Rt) for g in J] + [(1 - t) * f.set_ring(Rt)]
    # upstream groebner, so the prediction does not depend on the instrumented f5b
    survivors = [p for p in groebner(elim, Rt) if all(m[0] == 0 for m in p.monoms())]
    quotients = [s.set_ring(R).exquo(f) for s in survivors]
    lt_colon = [q.LM for q in groebner(quotients, R)] if quotients else []
    lt_J = [q.LM for q in groebner(J, R)]
    return sorted(minimalise(lt_colon)), sorted(minimalise(lt_J))


def predicted_zero_signatures(preprocessed, R):
    """{i: minimal generators of LT(J_i : f_i) not in LT(J_i)}, J_i = <f_{i+1}, ..., f_s>.

    Indices with no such generators are omitted.
    """
    predicted = {}
    for i in range(len(preprocessed), 0, -1):
        lt_colon, lt_J = colon_lt(preprocessed[i:], preprocessed[i - 1], R)
        monoms = [m for m in lt_colon if not any(monomial_divides(g, m) for g in lt_J)]
        if monoms:
            predicted[i] = monoms
    return predicted


def predict(run):
    prep = run.totals["preprocessed"]
    return prep, predicted_zero_signatures(prep, prep[0].ring)


@dataclass
class Decomposition:
    genuine: list = field(default_factory=list)
    redundant: list = field(default_factory=list)
    unexplained: list = field(default_factory=list)
    missed: list = field(default_factory=list)


def decompose_zero_reductions(zero_signatures, predicted):
    d = Decomposition()
    seen = set()
    for idx, m in zero_signatures:
        mon = tuple(m)
        if mon in predicted.get(idx, []) and (idx, mon) not in seen:
            d.genuine.append((idx, mon))
        elif any(g_idx == idx and monomial_divides(g_mon, mon) for g_idx, g_mon in d.genuine):
            d.redundant.append((idx, mon))
        else:
            d.unexplained.append((idx, mon))
        seen.add((idx, mon))
    d.missed = [(idx, mon) for idx, mons in predicted.items() for mon in mons
                if (idx, mon) not in seen]
    return d


def syzygy_degrees(predicted, preprocessed):
    out = {}
    for idx, mons in predicted.items():
        d_f = max(sum(m) for m in preprocessed[idx - 1].monoms())
        out[idx] = [(m, sum(m) + d_f) for m in mons]
    return out


def signature_lists(sigs):
    return [[i, list(m)] for i, m in sigs]


def decomposition_record(name, d, predicted, preprocessed):
    return {
        "system": name,
        "predicted": {str(i): [list(m) for m in v] for i, v in predicted.items()},
        "syzygy_degrees": {str(i): [[list(m), deg] for m, deg in v]
                           for i, v in syzygy_degrees(predicted, preprocessed).items()},
        "genuine": signature_lists(d.genuine),
        "redundant": signature_lists(d.redundant),
        "unexplained": signature_lists(d.unexplained),
        "missed": signature_lists(d.missed),
    }
