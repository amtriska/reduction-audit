import re
import shutil
import subprocess


def _poly_string(p):
    names = [str(s) for s in p.ring.symbols]
    terms = []
    for mon, c in p.terms():
        c = int(c) if p.ring.domain.characteristic() else c
        factors = [f"{n}^{e}" if e > 1 else n for n, e in zip(names, mon) if e]
        terms.append("*".join([f"({c})"] + factors))
    return " + ".join(terms) if terms else "0"


def singular_script(polys):
    R = polys[0].ring
    char = R.domain.characteristic()
    names = ",".join(str(s) for s in R.symbols)
    lines = [f"ring r = {char}, ({names}), dp;"]
    lines += [f"poly f{i} = {_poly_string(p)};" for i, p in enumerate(polys, start=1)]
    m = len(polys)
    for i in range(m - 1, 0, -1):
        J = ", ".join(f"f{j}" for j in range(i + 1, m + 1))
        lines += [
            f"ideal J = {J};",
            "ideal LJ = std(lead(std(J)));",
            f"ideal LQ = interred(lead(std(quotient(J, f{i}))));",
            "int k;",
            "for (k = 1; k <= ncols(LQ); k++) {",
            f'  if (LQ[k] != 0 and reduce(LQ[k], LJ) != 0) {{ print("SIG {i}:" + string(leadexp(LQ[k]))); }}',
            "}",
            "kill J, LJ, LQ, k;",
        ]
    lines.append("quit;")
    return "\n".join(lines) + "\n"


def singular_prediction(polys):
    exe = shutil.which("Singular")
    if exe is None:
        raise RuntimeError("Singular not found on PATH")
    out = subprocess.run([exe, "-q", "--no-rc"], input=singular_script(polys),
                         capture_output=True, text=True, check=True).stdout
    predicted = {}
    for idx, exps in re.findall(r"SIG (\d+):([\d,]+)", out):
        predicted.setdefault(int(idx), []).append(tuple(int(e) for e in exps.split(",")))
    return {k: sorted(v) for k, v in predicted.items()}
