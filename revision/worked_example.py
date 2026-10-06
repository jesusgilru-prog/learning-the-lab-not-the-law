"""Worked example (reviewers 1, 3, 4): re-run two benchmark data sets from their
recorded seeds and print the output of every decision node. The cell is fixed in
advance: high effect level, sigma=0.08, scale 1, F=4, replicate 0, for R3 and R4b."""
import json

import numpy as np

import audit_frozen as A
import benchmark_factorial as B

rows = [json.loads(l) for l in open("bench_K30.jsonl")]
pick = {}
for r in rows:
    if r["level"] == 1 and r["sigma"] == 0.08 and r["scale"] == 1 and r["F"] == 4 and r["rep"] == 0:
        pick[r["regime"]] = r
out = {}
for reg in ("R3", "R4b"):
    r = pick[reg]
    rng = np.random.default_rng(r["seed"])
    data = B.simulate(reg, 1, 0.08, 1, 4, rng)
    o = A.run_audit(*data, tau_phys=B.TAU_PHYS, rng=rng)
    ly, lre, mach, geom, fac = data
    m = o["m_hat"]
    low = mach < m
    facs, cnt_low = np.unique(fac[low], return_counts=True)
    o["n_low"], o["n"] = int(low.sum()), len(ly)
    o["low_by_fac"] = dict(zip(facs.tolist(), cnt_low.tolist()))
    hf, cnt_hi = np.unique(fac[~low], return_counts=True)
    o["max_share_high"] = float(cnt_hi.max() / (~low).sum())
    o["max_share_low"] = float(cnt_low.max() / low.sum())
    o["bf"] = float(np.exp(o["log_bf"]))
    o["bf_phys"] = float(np.exp(o["t2_logbf"]))
    same = (o["verdict_A"] == r["verdict_A"] and o["verdict_B"] == r["verdict_B"])
    print(reg, "reproduces stored run:", same)
    out[reg] = {k: v for k, v in o.items() if k != "t1_grid"}
    for k, v in out[reg].items():
        print(f"   {k}: {v}")
json.dump(out, open("worked_example.json", "w"), indent=1, default=str)
