from dataclasses import dataclass, field

from . import _vendored_sympy_gb as V


@dataclass
class Run:
    basis: list
    totals: dict
    events: list = field(default_factory=list)

    def same_basis_as(self, other):
        return set(map(str, self.basis)) == set(map(str, other.basis))


def run_f5b(system, syzygy_criterion=False, keep_zeros=False):
    events = []
    with V.instrumented(log=lambda **f: events.append(f), syzygy_criterion=syzygy_criterion,
                        keep_zeros=keep_zeros):
        G = V._f5b(list(system.polys), system.ring)
    return Run(G, V.last(), events)
