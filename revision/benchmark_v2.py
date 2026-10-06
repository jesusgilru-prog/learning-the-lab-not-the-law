"""Factorial synthetic benchmark for the frozen audit (reviewer 1, items on
component ablation and on the four-regime verdict benchmark).

Regimes (ground truth known by construction):
  R1  invariant law, no confounding
  R2  facility shift without structural confounding (facility-level offsets)
  R3  genuine physical regime transition at tau_phys (exponent switch in Mach,
      every facility samples both sides)
  R4a facility-induced pseudo-transition, disjoint band: one facility occupies its
      own Mach band and has its own exponent
  R4b facility-induced pseudo-transition, sparse support: a small low-Mach subset
      (~8% of points) drawn from two facilities carries an unmodelled additive
      artefact, mimicking the pooled support imbalance of the case study
Factors: effect level (low/high), noise sigma (0.03/0.08), sample-size scale (1x/2x),
number of facilities F (4/6). K replicates per cell.

Usage: python3 benchmark_factorial.py K N_WORKERS OUT.jsonl
"""
import itertools
import json
import sys
from multiprocessing import Pool

import numpy as np

import audit_v2 as A

TAU_PHYS = 0.20
Q_TRUE = -0.10
EFFECT = {  # (low, high)
    "R1": (0.0, 0.0),
    "R3w": (0.005, 0.015),  # weak genuine transition at tau_phys
    "R3s": (0.05, 0.15),    # genuine transition on sparse support (~8% below 0.11), reference correct
    "R3m": (0.05, 0.15),    # genuine transition at 0.25, reference given as 0.20 (misspecified)
    "R4c": (0.15, 0.40),    # artefact in one facility above 0.20, balanced support
    "R2": (0.15, 0.40),   # SD of facility offsets in log Cp
    "R3": (0.05, 0.15),   # exponent jump at tau_phys
    "R4a": (0.05, 0.15),  # exponent shift of the banded facility
    "R4b": (0.15, 0.40),  # additive artefact on the sparse low-Mach subset
}
FACILITIES = [  # name, n, n_geom, Re range
    ("A", 45, 5, (1e5, 1e7)),
    ("B", 41, 4, (5e5, 5e7)),
    ("C", 20, 2, (1e6, 8e7)),
    ("D", 8, 1, (3e5, 3e6)),
    ("E", 25, 2, (2e5, 2e7)),
    ("G", 12, 1, (1e6, 1e7)),
]


def simulate(regime, level, sigma, scale, F, rng):
    eff = EFFECT[regime][level]
    facs = FACILITIES[:F]
    ly, lre, mach, geom, fac = [], [], [], [], []
    offsets = {f[0]: (rng.normal(0, eff) if regime == "R2" else 0.0) for f in facs}
    for name, n, ngeom, (r0, r1) in facs:
        n = n * scale
        re = np.exp(rng.uniform(np.log(r0), np.log(r1), n))
        g = np.array([f"{name}{i % ngeom}" for i in range(n)])
        logC = {f"{name}{j}": -2.0 + rng.normal(0, 0.15) for j in range(ngeom)}
        if regime == "R4a" and name == "D":
            m = rng.uniform(0.40, 0.65, n)
        elif regime in ("R4b", "R3s"):
            m = rng.uniform(0.12, 0.35, n)
        else:
            m = rng.uniform(0.05, 0.35, n)
        q = np.full(n, Q_TRUE)
        add = np.full(n, offsets[name])
        if regime in ("R3", "R3w"):
            q = np.where(m >= TAU_PHYS, Q_TRUE + eff, Q_TRUE)
        if regime == "R3m":
            q = np.where(m >= 0.25, Q_TRUE + eff, Q_TRUE)
        if regime == "R3s":
            k = int(round(0.08 * n))
            idx = rng.choice(n, size=k, replace=False)
            m[idx] = rng.uniform(0.05, 0.10, k)
            q = np.where(m < 0.11, Q_TRUE, Q_TRUE + eff)
        if regime == "R4c" and name == "A":
            add = add + np.where(m >= TAU_PHYS, eff, 0.0)
        if regime == "R4a" and name == "D":
            q = q + eff
        if regime == "R4b" and name in ("A", "B"):
            k = int(round(0.04 * sum(f[1] for f in facs) * scale))
            idx = rng.choice(n, size=k, replace=False)
            m[idx] = rng.uniform(0.05, 0.10, k)
            add[idx] += eff
        y = np.array([logC[gg] for gg in g]) + q * np.log(re) + add + rng.normal(0, sigma, n)
        ly.append(y); lre.append(np.log(re)); mach.append(m); geom.append(g)
        fac.append(np.full(n, name))
    return (np.concatenate(ly), np.concatenate(lre), np.concatenate(mach),
            np.concatenate(geom), np.concatenate(fac))


def job(args):
    regime, level, sigma, scale, F, rep, seed = args
    rng = np.random.default_rng(seed)
    data = simulate(regime, level, sigma, scale, F, rng)
    tau = 0.11 if regime == "R3s" else TAU_PHYS
    out = A.run_audit(*data, tau_phys=tau, rng=rng)
    out.update(regime=regime, level=level, sigma=sigma, scale=scale, F=F, rep=rep,
               seed=seed, n=int(len(data[0])))
    return out


def main():
    K, workers, path = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    cells = list(itertools.product(EFFECT, (0, 1), (0.03, 0.08), (1, 2), (4, 6)))
    ss = np.random.SeedSequence(20261002)
    seeds = ss.generate_state(len(cells) * K)
    jobs = [(*c, r, int(seeds[i * K + r])) for i, c in enumerate(cells) for r in range(K)]
    with Pool(workers) as pool, open(path, "w") as fh:
        for i, out in enumerate(pool.imap_unordered(job, jobs, chunksize=1)):
            fh.write(json.dumps(out, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")
            fh.flush()
            if (i + 1) % 100 == 0:
                print(f"{i+1}/{len(jobs)}", flush=True)


if __name__ == "__main__":
    main()
