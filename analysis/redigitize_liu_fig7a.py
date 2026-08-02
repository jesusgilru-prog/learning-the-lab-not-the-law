import os
#!/usr/bin/env python
"""
Pixel-level re-digitisation of Figure 7(a) of Liu et al. (2024).

The original extraction (`hyperscale-chief/scripts/03_extract_liu.py`) read the
figure by eye on a 0.5 kW grid, which for the smallest bar (1.5 kW) is a ~17%
quantisation error -- far worse than the 5% the dataset declares.  This script
locates every marker to the pixel and calibrates against the axis ticks, giving
a resolution of ~0.06 kW.

Figure 7(a) is a dot plot, not a bar chart: grey markers are "Experiment",
red markers are "Estimation" (Eq. 12, P = C rho omega^3).  The two overlap in
the low-power rows, so marker centres are recovered from the unoccluded left
edge plus the measured marker radius.

Outputs `liu_fig7a_redigitized.csv` and prints a calibration report.
"""
import sys

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

IMG = "REDACTED: copyrighted source page image not redistributed here; supply your own extraction of Liu et al. 2024, Figure 7a (see paper Table A1 for the DOI)"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "liu_fig7a_redigitized.csv")

# Rows of the plot, top to bottom, as labelled on the y-axis.
ROWS = [
    (101, 1000), (101, 800), (50, 1000), (101, 600), (50, 800),
    (30, 1000), (101, 400), (50, 600), (30, 800), (30, 600),
    (50, 400), (101, 200), (30, 400), (50, 200), (30, 200),
]

# Table 4 of the paper: angular velocity per centrifugal acceleration.
OMEGA = {200: 44.27, 400: 62.61, 600: 76.68, 800: 88.54, 1000: 98.99}

# Air density used by the paper's own estimation curve.
R_GAS, M_AIR, T_AIR = 8.314, 0.029, 288.0

PLOT_Y = (1330, 2320)          # vertical extent of the plotting area
PLOT_X = (860, 2045)           # horizontal extent, inside the frame


def rho(p_kpa, T=T_AIR):
    return p_kpa * 1000.0 * M_AIR / (R_GAS * T)


def find_ticks(grey):
    """Axis ticks sit just below the bottom frame line."""
    dark = grey < 140
    ys = [y for y in range(1300, 2450) if dark[y, 850:2100].sum() > 900]
    bottom = max(ys)
    band = dark[bottom + 1: bottom + 21, 840:2060]
    col = band.sum(0)
    xs = np.nonzero(col >= 10)[0] + 840
    groups, cur = [], [xs[0]]
    for x in xs[1:]:
        if x - cur[-1] <= 4:
            cur.append(x)
        else:
            groups.append(cur)
            cur = [x]
    groups.append(cur)
    return np.array([np.mean(g) for g in groups]), bottom


def markers(mask, min_px=150):
    """Connected components of `mask` that look like a plot marker."""
    lab, n = ndimage.label(mask)
    out = []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(lab == i)
        if len(ys) < min_px:
            continue
        w, h = xs.max() - xs.min(), ys.max() - ys.min()
        if not (10 <= w <= 30 and 10 <= h <= 30):
            continue
        out.append(dict(cy=ys.mean(), cx=xs.mean(), xmin=xs.min(), xmax=xs.max(),
                        n=len(ys), w=w, h=h,
                        fill=len(ys) / (np.pi * (w / 2) * (h / 2))))
    return sorted(out, key=lambda d: d["cy"])


def main():
    rgb = np.array(Image.open(IMG).convert("RGB")).astype(int)
    grey_img = np.array(Image.open(IMG).convert("L")).astype(int)
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    ticks, bottom = find_ticks(grey_img)
    if len(ticks) != 7:
        print(f"ERROR: expected 7 axis ticks (0..60 kW), found {len(ticks)}: {ticks}")
        return 1
    tick_kw = np.arange(0, 70, 10)
    slope, intercept = np.polyfit(tick_kw, ticks, 1)          # px per kW
    resid = ticks - (slope * tick_kw + intercept)
    print("CALIBRACION DEL EJE X")
    print(f"  ticks (px): {np.round(ticks, 1)}")
    print(f"  {slope:.4f} px/kW  ->  resolucion de 1 px = {1/slope:.4f} kW")
    print(f"  residuos del ajuste lineal (px): {np.round(resid, 2)}  max={np.abs(resid).max():.2f}")
    to_kw = lambda x: (x - intercept) / slope

    inside = np.zeros(R.shape, bool)
    inside[PLOT_Y[0]:PLOT_Y[1], PLOT_X[0]:PLOT_X[1]] = True

    red = (R > 140) & (G < 110) & (B < 110) & ((R - G) > 60) & inside
    grey = (abs(R - G) < 20) & (abs(G - B) < 20) & (R > 45) & (R < 130) & inside

    # Keep round markers only (drops the italic "(a)" panel label) and discard
    # anything left of 0 kW (the legend swatches).
    def is_data(d):
        return 0.90 <= d["fill"] <= 1.30 and d["cx"] > ticks[0] + 5

    red_m = [d for d in markers(red) if is_data(d)]
    grey_m = [d for d in markers(grey) if d["cx"] > ticks[0] + 5]
    if len(red_m) != len(ROWS):
        print(f"ERROR: {len(red_m)} marcadores rojos, esperaba {len(ROWS)}")
        return 1

    # Marker radius from the cleanest (fully circular) red markers.
    radius = np.mean([(d["w"] + d["h"]) / 4 for d in red_m])
    print(f"\n  radio medio del marcador = {radius:.2f} px "
          f"({radius/slope:.3f} kW)")

    rows = []
    for (p_kpa, g_level), rm in zip(ROWS, red_m):
        y = rm["cy"]
        cand = [d for d in grey_m if abs(d["cy"] - y) < 18]
        if cand:
            gm = max(cand, key=lambda d: d["n"])
            occluded = gm["n"] < 300 or gm["w"] < 18
            # The experiment marker always sits left of the estimation marker,
            # so its left edge is never covered: reconstruct the centre from it.
            gx = gm["xmin"] + radius if occluded else gm["cx"]
            how = "borde izquierdo (ocluido)" if occluded else "centroide"
        else:
            # Fully hidden behind the red marker: search for grey pixels in the
            # row band regardless of blob size.
            band = grey[int(y) - 14: int(y) + 15, :]
            xs = np.nonzero(band.any(0))[0]
            if len(xs) == 0:
                gx, how = np.nan, "NO ENCONTRADO"
            else:
                gx, how = xs.min() + radius, "borde izquierdo (totalmente ocluido)"

        w = OMEGA[g_level]
        est_theory = 0.054 * rho(p_kpa) * w ** 3 / 1000.0
        rows.append(dict(pressure_kpa=p_kpa, g_level=g_level, omega_rad_s=w,
                         rho_kgm3=rho(p_kpa),
                         y_px=y, x_exp_px=gx, x_est_px=rm["cx"],
                         wrp_experiment_kW=to_kw(gx) if np.isfinite(gx) else np.nan,
                         wrp_estimation_kW=to_kw(rm["cx"]),
                         est_theory_kW=est_theory, method=how))
    df = pd.DataFrame(rows)

    print("\nVALIDACION: los marcadores rojos deben reproducir P = C rho omega^3")
    ok = df.dropna(subset=["wrp_estimation_kW"])
    c_fit = np.polyfit(ok.est_theory_kW, ok.wrp_estimation_kW, 1)
    pred = np.polyval(c_fit, ok.est_theory_kW)
    r2 = 1 - ((ok.wrp_estimation_kW - pred) ** 2).sum() / \
        ((ok.wrp_estimation_kW - ok.wrp_estimation_kW.mean()) ** 2).sum()
    # C implied by the digitised red markers
    c_implied = np.polyfit(ok.rho_kgm3 * ok.omega_rad_s ** 3 / 1000, ok.wrp_estimation_kW, 1)
    print(f"  leido = {c_fit[0]:.4f} * teorico(C=0.054) + {c_fit[1]:+.3f}   R2={r2:.6f}")
    print(f"  C implicito por los marcadores rojos = {c_implied[0]:.5f} "
          f"(el paper declara Ces=0.054)")

    print("\nRESULTADO (kW)")
    old = {(30, 200): 1.5, (50, 200): 2.5, (30, 400): 4.0, (101, 200): 5.0,
           (50, 400): 6.5, (30, 600): 7.5, (30, 800): 10.0, (50, 600): 11.0,
           (101, 400): 12.5, (30, 1000): 14.0, (50, 800): 16.0, (101, 600): 19.0,
           (50, 1000): 22.0, (101, 800): 32.0, (101, 1000): 53.0}
    df["wrp_old_kW"] = [old[(int(r.pressure_kpa), int(r.g_level))] for r in df.itertuples()]
    df["delta_pct"] = 100 * (df.wrp_experiment_kW / df.wrp_old_kW - 1)
    print(df[["pressure_kpa", "g_level", "wrp_experiment_kW", "wrp_old_kW",
              "delta_pct", "wrp_estimation_kW", "method"]]
          .to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
    print(f"\n  |delta| medio = {df.delta_pct.abs().mean():.1f}%   "
          f"max = {df.delta_pct.abs().max():.1f}%")

    df.to_csv(OUT, index=False)
    print(f"\nguardado: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
