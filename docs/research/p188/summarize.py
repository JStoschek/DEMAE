import json
import sys
from collections import defaultdict

import numpy as np

FORMS = ["flat", "peak", "ridge", "shoulder", "spur", "slope", "hollow", "footslope", "valley", "pit"]
d = json.load(open(sys.argv[1]))
recs = d["recs"]
agg = defaultdict(lambda: np.zeros(10))
flatstrata = defaultdict(list)
for r in recs:
    k = (r["L"], r["flat"], r["fdist"])
    c = np.array(r["counts"], float)
    agg[k + (r["band"],)] += c
    agg[k + ("all",)] += c
    if r["stratum"] in (2, 5, 7, 17):
        flatstrata[k].append(c[0] / max(c.sum(), 1))


def shares(k):
    c = agg[k]
    return c / c.sum()


base = (50, 1.0, "0")
print("config            | flat% per band 0..3      | rarest class% (band0) | absent classes per band | TV vs base (band3, all) | flat-strata tiles: median/max flat%")
for L in (25, 50):
    for fl in (1.0, 0.5, 0.2, 0.1):
        for fd in ("0", "half"):
            k = (L, fl, fd)
            fb = [100 * shares(k + (b,))[0] for b in range(4)]
            s0 = shares(k + (0,))
            rare = FORMS[int(np.argmin(s0))], 100 * s0.min()
            absent = [int((agg[k + (b,)] == 0).sum()) for b in range(4)]
            tv3 = 0.5 * np.abs(shares(k + (3,)) - shares(base + (3,))).sum()
            tva = 0.5 * np.abs(shares(k + ("all",)) - shares(base + ("all",))).sum()
            fs = np.array(flatstrata[k]) * 100
            print(f"L{L:<3} t{fl:<4} fd={fd:<5}| " + " ".join(f"{x:5.1f}" for x in fb)
                  + f" | {rare[0]:>9} {rare[1]:4.1f} | {absent} | {tv3:.3f} {tva:.3f} | {np.median(fs):5.1f} / {fs.max():5.1f}")
print()
for k in [(50, 1.0, "0"), (50, 0.2, "0"), (50, 0.1, "0"), (50, 0.2, "half"), (25, 0.2, "0")]:
    print(k)
    for b in list(range(4)) + ["all"]:
        print(f"  band {b}: " + " ".join(f"{n[:4]} {100*v:4.1f}" for n, v in zip(FORMS, shares(k + (b,)))))
