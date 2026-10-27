"""Rank renderer neurons Lyra-style from a sweep's measures (#6): python ace/renderer_rank.py runs/renderer_sweep_L12

For each measure and push direction, a unit's score is its average change across the plans. Random directions give
the "nothing special" spread of that score; a neuron's z is how many random-spreads it sits from the random average.
`agree` counts the plans whose change has the same sign as the average, so a high z from one song alone stands out.
Writes ranking.json next to measures.jsonl.
"""
import json
import os
import sys
from collections import defaultdict
from statistics import mean, pstdev


run = sys.argv[1]
rows = [json.loads(l) for l in open(os.path.join(run, "measures.jsonl"))]
plans = sorted({r["plan"] for r in rows})
MEASURES = [k for k in rows[0] if k not in ("plan", "kind", "unit", "dose", "file")]  # includes "moved"
by = defaultdict(dict)  # (kind, unit, dose) -> plan -> row
for r in rows:
    by[(r["kind"], r["unit"], r["dose"])][r["plan"]] = r

ranking = {}
for dose in sorted({r["dose"] for r in rows}):
    for m in MEASURES:
        scores = {}
        for (kind, unit, d), per_plan in by.items():
            if d == dose:
                changes = [per_plan[p][m] for p in plans]
                scores[(kind, unit)] = (mean(changes), changes)
        rand = [s for (kind, _), (s, _) in scores.items() if kind == "random"]
        mu, sd = mean(rand), pstdev(rand) or 1e-9
        max_rand = max(abs(s - mu) / sd for s in rand)
        units = []
        for (kind, unit), (s, changes) in scores.items():
            z = (s - mu) / sd
            agree = sum((c > 0) == (s > 0) for c in changes)
            units.append({"kind": kind, "unit": unit, "z": z, "mean": s, "changes": changes, "agree": agree})
        units.sort(key=lambda u: -abs(u["z"]))
        neurons = [u for u in units if u["kind"] == "neuron"]
        beyond = sum(abs(u["z"]) > max_rand for u in neurons)
        ranking[f"{m} {dose:+g}"] = {"measure": m, "dose": dose, "random_mean": mu, "random_sd": sd, "max_random_z": max_rand,
                                     "neurons_beyond_every_random": beyond, "n_neurons": len(neurons), "plans": plans,
                                     "units": units}
json.dump(ranking, open(os.path.join(run, "ranking.json"), "w"), indent=1)

print(f"{'measure':22} {'neurons beyond every random':>28}  top neurons (z, plans agreeing of {len(plans)})   N5455")
for key, r in ranking.items():
    top = [u for u in r["units"] if u["kind"] == "neuron"][:4]
    n5455 = next(u for u in r["units"] if u["kind"] == "neuron" and u["unit"] == 5455)
    rank5455 = [u["unit"] for u in r["units"] if u["kind"] == "neuron"].index(5455) + 1
    print(f"{key:22} {r['neurons_beyond_every_random']:>12} of {r['n_neurons']:<13}  "
          + "  ".join(f"N{u['unit']} {u['z']:+.1f} {u['agree']}/{len(plans)}" for u in top)
          + f"   #{rank5455} z {n5455['z']:+.1f}")
