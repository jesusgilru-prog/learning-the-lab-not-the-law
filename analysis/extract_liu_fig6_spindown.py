import os
#!/usr/bin/env python
"""Exact (vector-level) extraction of Liu 2024 Fig. 6 spin-down curves.

Unlike Fig. 7(a), Fig. 6 is stored as VECTOR graphics in the PDF: every marker
is a filled path with exact coordinates, so there is no pixel digitisation
error at all.  Axis calibration comes from the tick-label text positions
(also vector); linear-fit residuals are < 0.04 px on both axes.

Series: blue = omega(t) at 3 kPa, magenta = omega(t) at 10 kPa (legend order
on p. 18: "omega/3kPa Fit", "omega/10kPa Fit").  Two legend swatches plot at
omega > 1.09 rad/s and are removed (real data start at ~1.05 rad/s).

Output: liu_fig6_spindown.csv  (pressure_kpa, t_s, omega_rad_s)
"""
import sys

import fitz
import numpy as np
import pandas as pd

PDF = "REDACTED: copyrighted source PDF not redistributed here; supply your own copy of Liu et al. 2024 (see paper Table A1 for the DOI)"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "liu_fig6_spindown.csv")
PAGE = 18                                   # 0-based; Fig. 6 page


def main():
    pg = fitz.open(PDF)[PAGE]

    # -- axis calibration from tick-label word boxes ------------------------
    words = pg.get_text("words")
    xlab, ylab = {}, {}
    for x0, y0, x1, y1, t, *_ in words:
        if 660 < y0 < 690 and t in {"0", "20", "40", "60", "80"}:
            xlab[float(t)] = (x0 + x1) / 2
        if 180 < x0 < 200 and t in {"0.0", "0.2", "0.4", "0.6", "0.8", "1.0", "1.2"}:
            ylab[float(t)] = (y0 + y1) / 2
    assert len(xlab) == 5 and len(ylab) == 7, (xlab, ylab)

    tx = np.array(sorted(xlab)); px = np.array([xlab[k] for k in sorted(xlab)])
    wy = np.array(sorted(ylab)); py = np.array([ylab[k] for k in sorted(ylab)])
    mx, bx = np.polyfit(tx, px, 1)
    my, by = np.polyfit(wy, py, 1)
    rx = np.abs(px - (mx * tx + bx)).max()
    ry = np.abs(py - (my * wy + by)).max()
    print(f"calibracion t:     {mx:+.4f} px/s          residuo max {rx:.3f} px")
    print(f"calibracion omega: {my:+.4f} px/(rad/s)  residuo max {ry:.3f} px")

    # -- markers: filled vector paths, exact centres ------------------------
    series = {"blue": [], "magenta": []}
    for d in pg.get_drawings():
        if d["type"] != "fs":
            continue
        key = {(0.0, 0.0, 1.0): "blue", (1.0, 0.0, 1.0): "magenta"}.get(d["fill"])
        if key:
            r = d["rect"]
            series[key].append(((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2))

    rows = []
    for key, p_kpa in [("blue", 3), ("magenta", 10)]:
        pts = np.array(series[key])
        t = (pts[:, 0] - bx) / mx
        w = (pts[:, 1] - by) / my
        keep = w < 1.09                       # drops the legend swatch
        o = np.argsort(t[keep])
        for a, b in zip(t[keep][o], w[keep][o]):
            rows.append(dict(pressure_kpa=p_kpa, t_s=a, omega_rad_s=b))
        print(f"{key:8s} ({p_kpa:2d} kPa): {keep.sum()} puntos, "
              f"t=[{t[keep].min():.1f},{t[keep].max():.1f}] s, "
              f"omega=[{w[keep].min():.3f},{w[keep].max():.3f}] rad/s")

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"guardado: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
