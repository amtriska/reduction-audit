"""Regenerate the result files the note's tables are built from.

    python -m reduction_audit.records [--out runs]
"""
import argparse
import json
import os
import platform
import shutil
import subprocess
from datetime import datetime, timezone

import sympy
from sympy.polys.monomials import monomial_divides

from . import _vendored_sympy_gb as V
from .instances import ABLATION_SYSTEMS, DETERMINANTAL, STRUCTURED, SWEEP_SEEDS, SWEEP_SHAPES
from .run import run_f5b
from .systems import random_quadratic
from .syzygies import decompose_zero_reductions, decomposition_record, predict, signature_lists


def _decompose(run, pred):
    return decompose_zero_reductions(run.totals["zero_signatures"], pred)


def structured_record(system):
    plain = run_f5b(system)
    prep, pred = predict(plain)
    return decomposition_record(system.name, _decompose(plain, pred), pred, prep)


def determinantal_record(system, beta1, source):
    plain, syz = run_f5b(system), run_f5b(system, syzygy_criterion=True)
    prep, pred = predict(plain)
    dp, ds = _decompose(plain, pred), _decompose(syz, pred)
    record = decomposition_record(system.name, dp, pred, prep)
    record.update(beta1=beta1, beta1_source=source, redundant_count=len(dp.redundant),
                  same_basis=plain.same_basis_as(syz),
                  syz_clean=not (ds.unexplained or ds.missed or ds.redundant))
    return record


def sweep_record(n, m, seed):
    system = random_quadratic(n, m, seed)
    plain = run_f5b(system)
    syz = run_f5b(system, syzygy_criterion=True)
    kz = run_f5b(system, keep_zeros=True)
    prep, pred = predict(plain)
    dp, ds, dk = (_decompose(r, pred) for r in (plain, syz, kz))
    record = decomposition_record(system.name, dp, pred, prep)
    record.update(n=n, m=m, seed=seed,
                  same_basis=plain.same_basis_as(syz) and plain.same_basis_as(kz),
                  syz_clean=not (ds.unexplained or ds.missed or ds.redundant),
                  keep_zeros_redundant=signature_lists(dk.redundant),
                  keep_zeros_explained=not (dk.unexplained or dk.missed))
    return record


def _run_summary(run, d, plain):
    return {
        "zero_reductions": len(run.totals["zero_signatures"]),
        "genuine": len(d.genuine),
        "redundant": signature_lists(d.redundant),
        "unexplained": signature_lists(d.unexplained),
        "missed": signature_lists(d.missed),
        "sig_comparable": run.totals["sig_comparable"],
        "sig_rewritable": run.totals["sig_rewritable"],
        "sig_syzygy": run.totals["sig_syzygy"],
        "basis_size": len(run.basis),
        "same_basis_as_plain": plain.same_basis_as(run),
    }


def ablation_record(system):
    runs = {"plain": run_f5b(system),
            "keep_zeros": run_f5b(system, keep_zeros=True),
            "syzygy_criterion": run_f5b(system, syzygy_criterion=True)}
    _, pred = predict(runs["plain"])
    record = {"system": system.name, "n": system.ring.ngens, "m": len(system.polys),
              "predicted_total": sum(len(v) for v in pred.values())}
    for label, run in runs.items():
        record[label] = _run_summary(run, _decompose(run, pred), runs["plain"])
    return record


def trace_keep_zeros(system):
    """Run with zero results kept in B, logging each reduced pair's signature, the number of
    the pair's polynomial, whether it reduced to zero, and the number its result got in B."""
    log = []
    s_poly, f5_reduce = V.s_poly, V.f5_reduce

    def logged_s_poly(cp):
        log.append({"sig": (cp[0][1], tuple(cp[0][0])), "num": V.Num(cp[2])})
        return s_poly(cp)

    def logged_f5_reduce(f, B):
        p = f5_reduce(f, B)
        log[-1]["zero"] = not V.Polyn(p)
        return p

    V.s_poly, V.f5_reduce = logged_s_poly, logged_f5_reduce
    try:
        run = run_f5b(system, keep_zeros=True)
    finally:
        V.s_poly, V.f5_reduce = s_poly, f5_reduce
    # with keep_zeros every reduction result enters B, numbered consecutively after the inputs
    for k, e in enumerate(log, start=len(run.totals["preprocessed"]) + 1):
        e["result_num"] = k
    return run, log


def blind_spot_record(system):
    """For each redundant reduction left with zero results kept in B: the earlier zero results
    of the same index whose signatures divide its signature, and their numbers in B."""
    run, log = trace_keep_zeros(system)
    _, pred = predict(run)
    zeros = [e for e in log if e["zero"]]
    residuals = []
    for idx, mon in _decompose(run, pred).redundant:
        e = next(z for z in zeros if z["sig"] == (idx, mon))
        divisors = [z for z in zeros if z["result_num"] < e["result_num"] and z["sig"][0] == idx
                    and z["sig"] != e["sig"] and monomial_divides(z["sig"][1], mon)]
        residuals.append({
            "signature": [idx, list(mon)],
            "pair_polynomial_num": e["num"],
            "dividing_zero_results": [{"signature": [z["sig"][0], list(z["sig"][1])],
                                       "num": z["result_num"]} for z in divisors],
        })
    return {"system": system.name, "residuals": residuals}


def _git(*args):
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def _write(out, relpath, obj):
    path = os.path.join(out, relpath)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="runs")
    out = ap.parse_args().out

    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--", ".", f":(exclude){out}")
    provenance = {
        "commit": commit,
        "uncommitted_changes": bool(status) if commit else None,
        "sympy": sympy.__version__,
        "python": platform.python_version(),
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    # remove only what this script writes, so a stale file can't survive a regeneration
    for name in ("decompositions.json", "provenance.json"):
        if os.path.exists(os.path.join(out, name)):
            os.remove(os.path.join(out, name))
    for name in ("determinantal", "sweep", "ablations", "blind_spot"):
        shutil.rmtree(os.path.join(out, name), ignore_errors=True)

    _write(out, "decompositions.json", [structured_record(s) for s in STRUCTURED])
    for system, beta1, source in DETERMINANTAL:
        _write(out, f"determinantal/{system.name}.json",
               determinantal_record(system, beta1, source))
    for n, m in SWEEP_SHAPES:
        for seed in SWEEP_SEEDS:
            _write(out, f"sweep/n{n}m{m}s{seed}.json", sweep_record(n, m, seed))
    for system in ABLATION_SYSTEMS:
        _write(out, f"ablations/{system.name}.json", ablation_record(system))
        _write(out, f"blind_spot/{system.name}.json", blind_spot_record(system))
    _write(out, "provenance.json", provenance)


if __name__ == "__main__":
    main()
