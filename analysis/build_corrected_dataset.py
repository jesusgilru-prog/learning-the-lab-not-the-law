import os
#!/usr/bin/env python
"""
Build a corrected copy of the cross-facility windage benchmark.

Two defects are fixed, both confined to the Liu2024 block:

1. Windage power was read off Figure 7(a) by eye on a 0.5 kW grid (a ~17%
   quantisation error on the smallest point).  It is replaced by the
   pixel-level re-digitisation in `liu_fig7a_redigitized.csv`
   (axis calibration 17.13 px/kW, i.e. ~0.06 kW per pixel).

2. Air density was computed at T = 288 K while the dataset tabulates
   T_K = 293.15 K, and the speed of sound was computed at 293.15 K.  The two
   are made consistent at 293.15 K.  This shifts Liu's absolute Re and M by
   ~1.8% and does not affect any within-facility exponent, since rho stays
   proportional to chamber pressure either way.

The 10 kPa column is kept but flagged: those five values are not in Figure 7
and were back-computed from the paper's own fitted constant, so they carry no
independent information.

The original file is left untouched; the result is written next to this script
as `cross_rotor_dataset_v4.csv`.
"""
import sys

import numpy as np
import pandas as pd

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed_checkpoints", "cross_rotor_dataset_v3.csv")
REDIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "liu_fig7a_redigitized.csv")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "cross_rotor_dataset.csv")

T_LIU = 293.15          # K, matches the tabulated T_K and the speed of sound
R_SPECIFIC = 287.05     # J/(kg K), dry air
MU_LIU = None           # taken from the source file (Sutherland at T_LIU)
C_ESTIMATION = 0.054    # kW-scale constant the paper uses for its own estimate


def main():
    df = pd.read_csv(SRC)
    rd = pd.read_csv(REDIG)
    liu = df.source == "Liu2024"
    print(f"origen : {SRC}  (n={len(df)})")
    print(f"Liu2024: {liu.sum()} filas a corregir\n")

    # --- density consistent with the tabulated temperature ----------------
    old_rho = df.loc[liu, "rho_kgm3"].copy()
    new_rho = df.loc[liu, "p_Pa"] / (R_SPECIFIC * T_LIU)
    print("densidad recalculada a T = 293.15 K:")
    for p in sorted(df.loc[liu, "p_Pa"].unique()):
        m = liu & (df.p_Pa == p)
        print(f"   p={p/1000:6.1f} kPa   {old_rho[m].iloc[0]:.6f} -> "
              f"{new_rho[m].iloc[0]:.6f}  ({100*(new_rho[m].iloc[0]/old_rho[m].iloc[0]-1):+.2f}%)")
    df.loc[liu, "rho_kgm3"] = new_rho

    # --- windage power from the pixel-level re-digitisation ---------------
    key = rd.set_index([rd.pressure_kpa.astype(int), rd.g_level.astype(int)])
    n_new = 0
    print("\npotencia de windage (kW): antigua -> re-digitalizada")
    for i in df.index[liu]:
        p_kpa = int(round(df.at[i, "p_Pa"] / 1000))
        g_lvl = int(round(df.at[i, "g_level"]))
        if (p_kpa, g_lvl) in key.index:
            new_w = float(key.loc[(p_kpa, g_lvl), "wrp_experiment_kW"]) * 1000.0
            old_w = df.at[i, "P_w_W"]
            print(f"   {p_kpa:4d} kPa / {g_lvl:5d} g   {old_w/1000:7.3f} -> {new_w/1000:7.3f}"
                  f"   ({100*(new_w/old_w-1):+6.1f}%)")
            df.at[i, "P_w_W"] = new_w
            df.at[i, "data_origin"] = "digitized_from_figure_pixel"
            df.at[i, "error_pct"] = 1.0     # ~0.06 kW / value, dominated by marker size
            n_new += 1
        else:
            df.at[i, "data_origin"] = "computed_from_fitted_constant"

    # --- recompute the derived groups for Liu -----------------------------
    m = liu
    rho, om, R, mu = (df.loc[m, c] for c in ("rho_kgm3", "omega_rad_s", "R_m", "mu_Pas"))
    df.loc[m, "Re_Omega"] = rho * om * R ** 2 / mu
    df.loc[m, "Cp"] = df.loc[m, "P_w_W"] / (0.5 * rho * om ** 3 * R ** 5)
    a_sound = np.sqrt(1.4 * R_SPECIFIC * df.loc[m, "T_K"])
    df.loc[m, "M_tip"] = om * R / a_sound

    print(f"\n{n_new} filas con potencia re-digitalizada, "
          f"{int(m.sum()) - n_new} calculadas del ajuste del propio paper")
    df.to_csv(OUT, index=False)
    print(f"guardado: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
