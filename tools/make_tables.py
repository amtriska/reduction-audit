"""Build the note's tables (tables/*.tex) from runs/.

    python tools/make_tables.py
"""
import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs")
OUT = os.path.join(ROOT, "tables")

SYSTEMS = [  # (run name, label in the note); structured systems first, then determinantal
    ("cyclic-4", "cyclic-4"),
    ("katsura-3", "katsura-3"),
    ("katsura-4", "katsura-4"),
    ("trap", "trap"),
    ("twisted-cubic", "twisted cubic"),
    ("minors-2x3-generic", r"minors $2{\times}3$"),
    ("rnc-4", "RNC, degree 4"),
    ("minors-2x4-generic", r"minors $2{\times}4$"),
    ("minors2-3x3-generic", r"minors $3{\times}3$"),
]
LAST_STRUCTURED = "trap"


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def names(n):
    return ["x", "y", "z", "w", "u", "v"][:n] if n <= 6 else [f"x_{{{i}}}" for i in range(n)]


def monomial(exps):
    vs = names(len(exps))
    parts = [v if e == 1 else f"{v}^{{{e}}}" for v, e in zip(vs, exps) if e]
    return "".join(parts) if parts else "1"


def signature(idx, exps):
    return f"${monomial(exps)}\\,\\mathbf{{e}}_{{{idx}}}$"


def write(name, body):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as fh:
        fh.write(body)


def main_table(abl, det):
    rows = []
    for key, label in SYSTEMS:
        r = abl[key]
        lin = "---"
        if key in det:
            d = det[key]
            # all determinantal generators are quadrics, so a linear syzygy has degree 3
            linear = sum(1 for v in d["syzygy_degrees"].values() for _, deg in v if deg == 3)
            lin = f"{linear} / {d['beta1']}"
        rows.append(" & ".join([
            label, str(r["n"]), str(r["m"]), str(r["predicted_total"]),
            str(r["plain"]["zero_reductions"]), str(len(r["plain"]["redundant"])),
            str(len(r["keep_zeros"]["redundant"])), str(len(r["syzygy_criterion"]["redundant"])), lin]) + r" \\")
        if key == LAST_STRUCTURED:
            rows.append(r"\midrule")
    return "\n".join([
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r" & & & & & \multicolumn{3}{c}{redundant} & \\",
        r"\cmidrule(lr){6-8}",
        r"system & $n$ & $m$ & gen. & zeros & plain & A & B & lin./$\beta_1$ \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
    ]) + "\n"


def values(xs):
    return ", ".join(map(str, sorted(xs)))


def sweep_table(sweep):
    shapes = {}
    for r in sweep:
        shapes.setdefault((r["n"], r["m"]), []).append(r)
    rows = []
    for (n, m), rs in sorted(shapes.items()):
        g = {len(r["genuine"]) for r in rs}
        red = {len(r["redundant"]) for r in rs}
        a = {len(r["keep_zeros_redundant"]) for r in rs}
        explained = sum(1 for r in rs if not r["unexplained"] and not r["missed"])
        clean_b = sum(1 for r in rs if r["syz_clean"])
        sets = len({json.dumps(r["predicted"], sort_keys=True) for r in rs})
        rows.append(f"{n} & {m} & {len(rs)} & {values(g)} & {values(red)} & {values(a)} & "
                    f"{len(rs) - clean_b} & {explained}/{len(rs)} & {sets} \\\\")
    return "\n".join([
        r"\begin{tabular}{rrrrrrrrr}",
        r"\toprule",
        r" & & & & \multicolumn{3}{c}{redundant} & & pred. \\",
        r"\cmidrule(lr){5-7}",
        r"$n$ & $m$ & seeds & gen. & plain & A & B & explained & sets \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
    ]) + "\n"


def blind_spot_table(bs):
    rows = []
    for key, label in SYSTEMS:
        if key not in bs:
            continue
        for res in bs[key]["residuals"]:
            for z in res["dividing_zero_results"]:
                rows.append(" & ".join([label, signature(*res["signature"]),
                                        str(res["pair_polynomial_num"]), signature(*z["signature"]),
                                        str(z["num"])]) + r" \\")
    return "\n".join([
        r"\begin{tabular}{llrlr}",
        r"\toprule",
        r"system & residual & $\mathrm{Num}(\mathcal{F})$ & divided by & $\mathrm{Num}$ \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
    ]) + "\n"


def criteria_table(abl, key="cyclic-4"):
    r = abl[key]
    rows = [f"{lab} & {r[k]['zero_reductions']} & {r[k]['sig_comparable']} & {r[k]['sig_rewritable']} & "
            f"{r[k]['sig_syzygy']} & {r[k]['basis_size']} \\\\"
            for k, lab in (("plain", "plain"), ("keep_zeros", "A"),
                           ("syzygy_criterion", "B"))]
    return "\n".join([
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"run & zero red. & comparable & rewritable & syzygy & basis \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
    ]) + "\n"


def appendix(dec, det):
    records = {**dec, **det}
    out = []
    for key, label in SYSTEMS:
        r = records[key]
        if not r["predicted"]:
            continue
        degs = {(int(i), tuple(m)): d for i, v in r["syzygy_degrees"].items() for m, d in v}
        genuine = {(i, tuple(m)) for i, m in r["genuine"]}
        out.append(f"\\subsubsection*{{{label}}}")
        out.append(r"\begin{tabular}{lrl}")
        out.append(r"predicted (= genuine) & degree & observed in plain run \\")
        out.append(r"\hline")
        for (i, m), d in sorted(degs.items(), key=lambda t: (-t[0][0], t[1], t[0][1])):
            seen = "genuine" if (i, m) in genuine else "\\textbf{missed}"
            out.append(f"{signature(i, m)} & {d} & {seen} \\\\")
        for i, m in r["redundant"]:
            out.append(f"{signature(i, m)} & & redundant \\\\")
        out.append(r"\end{tabular}" + "\n")
    return "\n".join(out)


def sweep_appendix(sweep):
    rows = []
    for r in sorted(sweep, key=lambda r: (r["n"], r["m"], r["seed"])):
        rows.append(f"{r['n']} & {r['m']} & {r['seed']} & {len(r['genuine'])} & {len(r['redundant'])} & "
                    f"{len(r['keep_zeros_redundant'])} & {'yes' if r['syz_clean'] else 'no'} \\\\")
    return "\n".join([
        r"\begin{longtable}{rrrrrrr}",
        r"\toprule",
        r"$n$ & $m$ & seed & genuine & red.\ (plain) & red.\ (A) & B clean \\",
        r"\midrule\endhead",
        *rows,
        r"\bottomrule",
        r"\end{longtable}",
    ]) + "\n"


def main():
    dec = {r["system"]: r for r in load(os.path.join(RUNS, "decompositions.json"))}
    det = {os.path.basename(p)[:-5]: load(p) for p in glob.glob(os.path.join(RUNS, "determinantal", "*.json"))}
    abl = {os.path.basename(p)[:-5]: load(p) for p in glob.glob(os.path.join(RUNS, "ablations", "*.json"))}
    bs = {os.path.basename(p)[:-5]: load(p) for p in glob.glob(os.path.join(RUNS, "blind_spot", "*.json"))}
    sweep = [load(p) for p in glob.glob(os.path.join(RUNS, "sweep", "*.json"))]
    write("main.tex", main_table(abl, det))
    write("sweep.tex", sweep_table(sweep))
    write("blind_spot.tex", blind_spot_table(bs))
    write("criteria_cyclic4.tex", criteria_table(abl))
    write("appendix_signatures.tex", appendix(dec, det))
    write("appendix_sweep.tex", sweep_appendix(sweep))


if __name__ == "__main__":
    main()
