from .systems import cyclic, katsura, minors_2x2, random_quadratic, trap, twisted_cubic

STRUCTURED = [
    cyclic(4),
    katsura(3),
    katsura(4),
    trap(),
    random_quadratic(3, 4, 1),
    random_quadratic(3, 4, 2),
    random_quadratic(3, 5, 1),
    random_quadratic(4, 5, 1),
    twisted_cubic(),
]

DETERMINANTAL = [  # (system, beta_1, source of beta_1)
    (twisted_cubic(), 2, "Hilbert-Burch"),
    (minors_2x2([[0, 1, 2], [3, 4, 5]], 6, "minors-2x3-generic"), 2, "Eagon-Northcott"),
    (minors_2x2([[0, 1, 2, 3], [1, 2, 3, 4]], 5, "rnc-4"), 8, "Eagon-Northcott"),
    (minors_2x2([[0, 1, 2, 3], [4, 5, 6, 7]], 8, "minors-2x4-generic"), 8, "Eagon-Northcott"),
    (minors_2x2([[0, 1, 2], [3, 4, 5], [6, 7, 8]], 9, "minors2-3x3-generic"), 16, "Segre P2xP2"),
]

SWEEP_SHAPES = [(3, 4), (3, 5), (3, 6), (4, 5), (4, 6), (5, 6)]  # (n, m)
SWEEP_SEEDS = range(1, 11)

# structured and determinantal systems, each once (twisted-cubic is in both lists)
ABLATION_SYSTEMS = list({s.name: s for s in STRUCTURED + [s for s, *_ in DETERMINANTAL]}.values())


def declared_instances():
    """The 69 declared instances, each once (some structured entries are also sweep instances)."""
    sweep = [random_quadratic(n, m, s) for n, m in SWEEP_SHAPES for s in SWEEP_SEEDS]
    return list({s.name: s for s in ABLATION_SYSTEMS + sweep}.values())
