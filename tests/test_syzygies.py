import functools
import inspect
import shutil
from collections import Counter

import pytest
import sympy
import sympy.polys.groebnertools as gt

from reduction_audit import _vendored_sympy_gb as V
from reduction_audit.instances import (ABLATION_SYSTEMS, DETERMINANTAL, STRUCTURED, SWEEP_SEEDS,
                                       SWEEP_SHAPES, declared_instances)
from reduction_audit.records import (ablation_record, blind_spot_record, determinantal_record,
                                     structured_record, sweep_record)
from reduction_audit.run import run_f5b
from reduction_audit.singular import singular_prediction
from reduction_audit.systems import cyclic, katsura, random_quadratic, trap, twisted_cubic
from reduction_audit.syzygies import (decompose_zero_reductions, predict,
                                      predicted_zero_signatures, syzygy_degrees)


def as_sets(d):
    return {k: set(v) for k, v in d.items()}


def by_index(sigs):
    out = {}
    for idx, mon in sigs:
        out.setdefault(idx, set()).add(tuple(mon))
    return out


def as_tuples(sig_lists):
    return [(i, tuple(m)) for i, m in sig_lists]


def degrees(polys):
    return [max(sum(m) for m in f.monoms()) for f in polys]


def name_of(system):
    return system.name


STRUCTURED_EXPECTED = {  # name: (predicted total, redundant in the plain run)
    "cyclic-4": (3, [(2, (0, 0, 2, 2)), (2, (0, 0, 1, 5))]),
    "katsura-3": (0, []),
    "katsura-4": (0, []),
    "trap": (0, []),
    "mq-3-4-s1-p32003": (3, []),
    "mq-3-4-s2-p32003": (3, []),
    "mq-3-5-s1-p32003": (5, [(2, (0, 0, 1))]),
    "mq-4-5-s1-p32003": (6, [(2, (3, 1, 0, 0))]),
    "twisted-cubic": (2, []),
}

DETERMINANTAL_EXPECTED = {  # name: (predicted total, syzygy-degree counts)
    "twisted-cubic": (2, {3: 2}),
    "minors-2x3-generic": (2, {3: 2}),
    "rnc-4": (8, {3: 8}),
    "minors-2x4-generic": (8, {3: 8}),
    "minors2-3x3-generic": (17, {3: 16, 4: 1}),
}

SWEEP_EXPECTED = {  # (n, m): (genuine, redundant) in the plain run
    (3, 4): (3, 0), (3, 5): (5, 1), (3, 6): (8, 1),
    (4, 5): (6, 1), (4, 6): (8, 2), (5, 6): (10, 8),
}

KEEP_ZEROS = {  # name: (redundant in the plain run, redundant with zero results kept in B)
    "cyclic-4": (2, 1), "katsura-3": (0, 0), "katsura-4": (0, 0), "trap": (0, 0),
    "mq-3-4-s1-p32003": (0, 0), "mq-3-4-s2-p32003": (0, 0), "mq-3-5-s1-p32003": (1, 0),
    "mq-4-5-s1-p32003": (1, 0), "twisted-cubic": (0, 0), "minors-2x3-generic": (0, 0),
    "rnc-4": (5, 1), "minors-2x4-generic": (2, 1), "minors2-3x3-generic": (12, 4),
}
ABLATION_BY_NAME = {s.name: s for s in ABLATION_SYSTEMS}

# Lines of upstream groebnertools.py that fix the order in which _f5b selects pairs,
# with their line numbers in SymPy 1.14.0.
F5B_ORDER_LINES = {
    "sig_key": ("return (-s[1], order(s[0]))", [334]),
    "cp_key": ("return (lbp_key(lbp(c[0], ring.zero, Num(c[2]))), "
               "lbp_key(lbp(c[3], ring.zero, Num(c[5]))))", [481]),
    "sort": ("CP.sort(key=lambda cp: cp_key(cp, ring), reverse=True)", [629, 674]),
    "pop": ("cp = CP.pop()", [636]),
}


@pytest.mark.parametrize("system", [cyclic(4), katsura(3), trap(), random_quadratic(3, 5, 1),
                                    random_quadratic(4, 6, 2), twisted_cubic()], ids=name_of)
def test_vendored_f5b_matches_upstream_with_options_off(system):
    assert run_f5b(system).basis == gt._f5b(list(system.polys), system.ring)


@pytest.mark.parametrize("system", STRUCTURED, ids=name_of)
def test_syz_run_reduces_exactly_the_predicted_signatures(system):
    run = run_f5b(system, syzygy_criterion=True)
    _, pred = predict(run)
    assert by_index(run.totals["zero_signatures"]) == as_sets(pred)
    assert sum(len(v) for v in pred.values()) == STRUCTURED_EXPECTED[system.name][0]


def test_pinned_signature_sets():
    prep, pred = predict(run_f5b(cyclic(4), syzygy_criterion=True))
    assert as_sets(pred) == {2: {(0, 0, 2, 0), (0, 0, 1, 4), (0, 1, 0, 5)}}
    assert sorted(deg for _, deg in syzygy_degrees(pred, prep)[2]) == [4, 7, 8]
    _, pred = predict(run_f5b(random_quadratic(3, 4, 1), syzygy_criterion=True))
    assert as_sets(pred) == {1: {(0, 0, 1), (0, 1, 0)}, 2: {(1, 1, 0)}}
    prep, pred = predict(run_f5b(twisted_cubic(), syzygy_criterion=True))
    assert as_sets(pred) == {1: {(0, 0, 0, 1), (0, 0, 1, 0)}}
    assert syzygy_degrees(pred, prep) == {1: [((0, 0, 0, 1), 3), ((0, 0, 1, 0), 3)]}


@pytest.mark.parametrize("system", STRUCTURED, ids=name_of)
def test_plain_run_decomposes_with_nothing_unexplained(system):
    record = structured_record(system)
    assert as_tuples(record["redundant"]) == STRUCTURED_EXPECTED[system.name][1]
    assert record["unexplained"] == []
    assert record["missed"] == []


def test_intervention_removes_exactly_the_redundant_reductions():
    plain, syz = run_f5b(cyclic(4)), run_f5b(cyclic(4), syzygy_criterion=True)
    assert plain.same_basis_as(syz)
    assert len(plain.basis) == 7
    assert plain.totals["sig_comparable"] == syz.totals["sig_comparable"] == 21
    assert plain.totals["sig_rewritable"] == syz.totals["sig_rewritable"] == 5
    assert plain.totals["sig_syzygy"] == 0
    assert syz.totals["sig_syzygy"] == 2
    _, pred = predict(plain)
    d = decompose_zero_reductions(plain.totals["zero_signatures"], pred)
    rejected = [(e["sig_index"], e["signature"]) for e in syz.events
                if e["rejected_by"] == "syzygy"]
    assert rejected == d.redundant
    assert syz.totals["zero_signatures"] == d.genuine


def test_predictor_requires_preprocessed_generators():
    for system in [cyclic(4), katsura(3), random_quadratic(3, 4, 1), random_quadratic(3, 5, 1),
                   random_quadratic(4, 5, 1), twisted_cubic()]:
        prep = run_f5b(system).totals["preprocessed"]
        assert len(prep) == len(system.polys), system.name
        assert degrees(prep) == degrees(system.polys), system.name
    system = cyclic(4)
    run = run_f5b(system, syzygy_criterion=True)
    _, pred = predict(run)
    raw = predicted_zero_signatures(system.polys, system.ring)
    assert raw == {1: [(0, 0, 2, 2), (0, 1, 0, 0), (1, 0, 0, 0)]}
    assert as_sets(raw) != as_sets(pred)
    assert as_sets(raw) != by_index(run.totals["zero_signatures"])


def test_regular_controls_predict_nothing():
    for system in [katsura(3), katsura(4), trap()]:
        plain, syz = run_f5b(system), run_f5b(system, syzygy_criterion=True)
        assert plain.totals["zero_signatures"] == [], system.name
        assert syz.totals["zero_signatures"] == [], system.name
        assert predict(plain)[1] == {}, system.name


def test_decomposition_set_logic():
    d = decompose_zero_reductions(
        [(1, (0, 3)), (1, (1, 0)), (1, (2, 0)), (2, (1, 0)), (1, (1, 0))],
        {1: [(1, 0), (0, 1), (0, 2)]})
    assert d.genuine == [(1, (1, 0))]
    assert d.redundant == [(1, (2, 0)), (1, (1, 0))]
    assert d.unexplained == [(1, (0, 3)), (2, (1, 0))]
    assert d.missed == [(1, (0, 1)), (1, (0, 2))]


def reduced_pair_signatures(system, syzygy_criterion, monkeypatch):
    seen = []
    original = V.s_poly

    # _f5b calls s_poly only for pairs that survived every criterion
    def s_poly(cp):
        seen.append(cp[0])
        return original(cp)

    with monkeypatch.context() as mp:
        mp.setattr(V, "s_poly", s_poly)
        run_f5b(system, syzygy_criterion=syzygy_criterion)
    return seen


def test_f5b_reduces_pairs_in_signature_order(monkeypatch):
    assert sympy.__version__ == "1.14.0"
    upstream = inspect.getsource(gt).splitlines()
    vendored = {line.strip() for line in inspect.getsource(V).splitlines()}
    for name, (text, numbers) in F5B_ORDER_LINES.items():
        assert [i + 1 for i, line in enumerate(upstream) if line.strip() == text] == numbers, name
        assert text in vendored, name
    systems = [cyclic(4), katsura(3), trap(), random_quadratic(3, 5, 1),
               random_quadratic(4, 6, 1)] + [s for s, *_ in DETERMINANTAL]
    for system in systems:
        for syz in (False, True):
            sigs = reduced_pair_signatures(system, syz, monkeypatch)
            keys = [V.sig_key(s, system.ring.order) for s in sigs]
            assert sigs, (system.name, syz)
            assert all(a <= b for a, b in zip(keys, keys[1:])), (system.name, syz)


@pytest.mark.parametrize("system, beta1, source", DETERMINANTAL,
                         ids=[s.name for s, *_ in DETERMINANTAL])
def test_determinantal_ideals(system, beta1, source):
    record = determinantal_record(system, beta1, source)
    total, degree_counts = DETERMINANTAL_EXPECTED[system.name]
    assert record["unexplained"] == []
    assert record["missed"] == []
    assert record["syz_clean"]
    assert record["same_basis"]
    assert sum(len(v) for v in record["predicted"].values()) == total
    counts = Counter(deg for v in record["syzygy_degrees"].values() for _, deg in v)
    assert counts == degree_counts
    assert counts[3] == beta1


@functools.cache
def cached_sweep_record(n, m, seed):
    return sweep_record(n, m, seed)


def slow_marks(n, m):
    return [pytest.mark.slow] if (n, m) == (5, 6) else []


@pytest.mark.parametrize("n, m, seed", [
    pytest.param(n, m, s, id=f"n{n}m{m}s{s}", marks=slow_marks(n, m))
    for n, m in SWEEP_SHAPES for s in SWEEP_SEEDS])
def test_random_quadratic_sweep(n, m, seed):
    record = cached_sweep_record(n, m, seed)
    assert record["unexplained"] == []
    assert record["missed"] == []
    assert record["syz_clean"]
    assert record["same_basis"]
    assert record["keep_zeros_explained"]
    assert record["keep_zeros_redundant"] == []
    assert (len(record["genuine"]), len(record["redundant"])) == SWEEP_EXPECTED[(n, m)]


@pytest.mark.parametrize("n, m", [
    pytest.param(n, m, id=f"n{n}m{m}", marks=slow_marks(n, m)) for n, m in SWEEP_SHAPES])
def test_sweep_prediction_does_not_depend_on_seed(n, m):
    predicted = [cached_sweep_record(n, m, s)["predicted"] for s in SWEEP_SEEDS]
    assert predicted == [predicted[0]] * len(predicted)


@functools.cache
def cached_ablation_record(name):
    return ablation_record(ABLATION_BY_NAME[name])


@pytest.mark.parametrize("name", list(KEEP_ZEROS))
def test_keep_zeros_restores_the_cited_algorithm(name):
    # Sun-Wang's F5B adds every reduction result to B, zero or not; upstream SymPy drops zeros
    record = cached_ablation_record(name)
    plain, kz = record["plain"], record["keep_zeros"]
    assert kz["same_basis_as_plain"]
    assert kz["genuine"] == plain["genuine"]
    assert kz["unexplained"] == []
    assert kz["missed"] == []
    assert (len(plain["redundant"]), len(kz["redundant"])) == KEEP_ZEROS[name]
    assert kz["sig_syzygy"] == 0


@pytest.mark.parametrize("name", list(KEEP_ZEROS))
def test_syzygy_criterion_leaves_only_the_genuine_reductions(name):
    record = cached_ablation_record(name)
    syz = record["syzygy_criterion"]
    assert syz["same_basis_as_plain"]
    assert syz["genuine"] == record["predicted_total"]
    assert syz["redundant"] == []
    assert syz["unexplained"] == []
    assert syz["missed"] == []


@pytest.mark.parametrize("name", list(KEEP_ZEROS))
def test_keep_zeros_residuals_are_the_rewrite_rule_blind_spot(name):
    # Each residual's signature is a multiple of an earlier zero's, but the pair's polynomial
    # is newer than that zero, so the rewritable criterion (num < Num(h)) cannot discard it.
    residuals = blind_spot_record(ABLATION_BY_NAME[name])["residuals"]
    assert len(residuals) == KEEP_ZEROS[name][1]
    for r in residuals:
        assert r["dividing_zero_results"], r["signature"]
        assert all(r["pair_polynomial_num"] > z["num"] for z in r["dividing_zero_results"]), \
            r["signature"]


def test_declared_instance_set_has_69_systems():
    assert len(declared_instances()) == 69


needs_singular = pytest.mark.skipif(shutil.which("Singular") is None,
                                    reason="Singular not installed")


def singular_param(system):
    slow = system.name.startswith("mq-5-6-")
    return pytest.param(system, id=system.name, marks=[pytest.mark.slow] if slow else [])


@needs_singular
@pytest.mark.parametrize("system", [singular_param(s) for s in declared_instances()])
def test_predictor_agrees_with_singular(system):
    # independent of SymPy: Singular's own quotient, std and lead on the same sequence
    prep, pred = predict(run_f5b(system))
    assert singular_prediction(prep) == {k: sorted(v) for k, v in pred.items()}


@needs_singular
def test_singular_check_can_fail():
    system = cyclic(4)
    _, pred = predict(run_f5b(system))
    assert singular_prediction(system.polys) != {k: sorted(v) for k, v in pred.items()}
