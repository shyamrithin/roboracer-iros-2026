#!/usr/bin/env python3
# =============================================================================
# heading_fit.py
# -----------------------------------------------------------------------------
# Project : RoboRacer Sim Racing League @ IROS 2026 - Team CEM Navigators (26)
# Author  : Shyam Rithin
# Location: ~/roboracer/tools/heading_fit.py
# Usage   : python3 ~/roboracer/tools/heading_fit.py "~/mapdata/V1 Log.csv"
# =============================================================================
#
# DESCRIPTION
# -----------
# Identifies which column of the AutoDRIVE data-recorder CSV actually carries
# vehicle yaw, in which sign convention, and with what constant offset --
# WITHOUT building a map.
#
# The idea: the RoboRacer is non-holonomic and (in these logs) always moving
# forward, so its true heading is fully determined by the ground-truth
# positions alone:
#
#       theta_true(t) = atan2( y[t+k] - y[t] ,  x[t+k] - x[t] )
#
# That is an independent measurement of heading that owes nothing to the euler
# columns. We then fit each candidate euler column against it:
#
#       theta_csv * s  ==  theta_true + theta0        (s = +1 or -1)
#
# and report the circular residual spread. The correct (column, sign) pair
# should collapse to a residual std of a few hundredths of a radian. Wrong
# pairs stay near the ~1.8 rad std of a uniform circle. This separates
# candidates by an order of magnitude rather than the 12% margin that the
# occupied-cell-count scorer gave.
#
# WHY THIS MATTERS FOR THE SMEARED MAP
# ------------------------------------
# A constant heading error does NOT rigidly rotate the accumulated map. It
# rotates every scan about its own pose, which is a different rigid transform
# at each sample. The result is exactly the observed failure mode: walls
# smeared 20-30 cm, error growing with range, and an inflated short axis
# (~21 x 31 m for a track that is ~30 x 10 m).
#
# WHAT IT CHECKS
# --------------
#   1. Per-column statistics for parts[5..11] -- a real yaw channel must sweep
#      the full circle over a lap; a near-constant column is an axis value.
#   2. Whether parts[7] is near-constant (Unity is Y-up; if the vertical axis
#      is being used as a ground-plane coordinate the projection is wrong).
#   3. All 6 candidates: {3 euler columns} x {2 signs}, each fitted for its
#      own best constant offset.
#   4. Residual stationarity -- residuals binned over time, so a constant
#      offset (fixable) is distinguished from a drifting one (not fixable by
#      a single constant).
#
# OUTPUT
# ------
# A ranked table. Read the top row. If its residual std is < ~0.05 rad and the
# per-bin offsets in the stationarity table agree to within a few hundredths,
# the projection fix is: use that column, that sign, that position frame, and
# subtract that theta0. If nothing gets below ~0.3 rad, the heading model is
# not the problem and we stop rather than tuning around it.
#
# Read-only. Touches no other file, writes nothing, modifies no map code.
# =============================================================================

import sys
import os
import math
import numpy as np

# ---- CSV layout (AutoDRIVE data recorder: no header row, 22 fields) --------
COL_POS_X = 5
COL_POS_Y = 6
COL_POS_Z = 7
EULER_COLS = [9, 10, 11]

# ---- Fit settings ----------------------------------------------------------
LOOKAHEAD_K = 3          # half-width of the centred position difference
MIN_DISP_M = 0.05        # reject samples where the car is essentially stopped
N_TIME_BINS = 8          # bins for the stationarity check
JUMP_M = 1.0             # a step larger than this between consecutive rows
                         # means a reset or a spliced recording


def wrap_pi(a):
    """Wrap angle(s) to (-pi, pi]."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def circ_mean(a):
    """Circular mean of an array of angles."""
    return math.atan2(np.mean(np.sin(a)), np.mean(np.cos(a)))


def circ_std(a):
    """
    Circular standard deviation. Returns ~1.8 rad for uniformly scattered
    angles, ~0 for tightly clustered ones.
    """
    R = math.hypot(np.mean(np.sin(a)), np.mean(np.cos(a)))
    R = min(max(R, 1e-12), 1.0)
    return math.sqrt(-2.0 * math.log(R))


def load(path):
    """Load the numeric columns we care about, tolerating short/blank rows."""
    xs, ys, zs, eul = [], [], [], []
    n_total = n_bad = 0
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            n_total += 1
            parts = line.split(",")
            if len(parts) <= max(EULER_COLS):
                n_bad += 1
                continue
            try:
                xs.append(float(parts[COL_POS_X]))
                ys.append(float(parts[COL_POS_Y]))
                zs.append(float(parts[COL_POS_Z]))
                eul.append([float(parts[c]) for c in EULER_COLS])
            except ValueError:
                n_bad += 1
                continue
    if n_bad:
        print(f"  (skipped {n_bad} of {n_total} unparseable rows)")
    return (np.asarray(xs), np.asarray(ys), np.asarray(zs),
            np.asarray(eul))


def column_stats(x, y, z, eul):
    print("\n" + "=" * 74)
    print("1. COLUMN STATISTICS")
    print("=" * 74)
    print(f"{'col':>5} {'name':<10} {'min':>10} {'max':>10} "
          f"{'range':>10} {'std':>10}")
    print("-" * 74)
    rows = [(COL_POS_X, "posX", x), (COL_POS_Y, "posY", y),
            (COL_POS_Z, "posZ", z)]
    for i, c in enumerate(EULER_COLS):
        rows.append((c, f"euler[{c}]", eul[:, i]))
    for c, name, v in rows:
        print(f"{c:>5} {name:<10} {v.min():>10.4f} {v.max():>10.4f} "
              f"{v.max() - v.min():>10.4f} {v.std():>10.4f}")

    print("\n  Interpretation:")
    for c, name, v in rows[3:]:
        rng = v.max() - v.min()
        if rng < 0.5:
            verdict = "near-CONSTANT -> not yaw (fixed axis / level vehicle)"
        elif rng > 5.5:
            verdict = "sweeps ~full circle -> YAW CANDIDATE"
        elif rng > 2.5:
            verdict = "partial sweep -> possible yaw, or wrapped differently"
        else:
            verdict = "limited range -> unlikely yaw (roll/pitch)"
        print(f"    {name:<10} range {rng:6.3f} rad  {verdict}")

    zr = z.max() - z.min()
    print(f"\n    posZ range {zr:.4f} m", end="  ")
    if zr < 0.05:
        print("-> flat: parts[5],[6] are a sensible ground plane.")
    else:
        print("-> VARIES. Check axis order: Unity is Y-up, so the "
              "ground plane\n       may not be parts[5],[6]. "
              "Investigate before trusting the map.")

    for c, name, v in rows[3:]:
        if v.max() > 7.0 or v.min() < -7.0:
            print(f"\n    NOTE: {name} exceeds +/-2pi -- likely DEGREES, "
                  "not radians.")


def build_truth(x, y, k, min_disp, jump_m):
    """
    Heading from ground-truth positions alone, using a CENTRED difference:

        theta_true[t] = atan2( y[t+k] - y[t-k] , x[t+k] - x[t-k] )

    Centred, not forward: a forward difference measures the heading at the
    midpoint of the interval while we compare it against the euler value at
    the start, which biases the recovered offset by half the yaw rate times
    the window. Centring removes that term exactly.

    Only one position frame is tested. atan2(dx,dy) == pi/2 - atan2(dy,dx),
    so a swapped frame is algebraically identical to a sign flip plus an
    offset -- it carries no independent information. And since theta_true is
    derived from the SAME positions that build_map.py projects with, any
    consistent relabelling of the axes cancels. The fit therefore answers
    precisely the question the map needs: given parts[5],[6] as the ground
    plane, which heading channel makes the vehicle motion consistent.

    Returns (theta_true, centre_indices, displacements, n_jumps).
    """
    n = len(x)
    t = np.arange(k, n - k)
    dx = x[t + k] - x[t - k]
    dy = y[t + k] - y[t - k]
    disp = np.hypot(dx, dy)

    # Discontinuity guard: the recorder appends if you record twice without
    # resetting, so a log may splice two sessions. A splice shows up as an
    # impossibly large step between consecutive samples.
    step = np.hypot(np.diff(x), np.diff(y))
    n_jumps = int(np.sum(step > jump_m))

    keep = (disp > min_disp) & (disp < jump_m * 2 * k)
    return np.arctan2(dy[keep], dx[keep]), t[keep], disp, n_jumps


def main():
    if len(sys.argv) < 2:
        print(__doc__ or "")
        print("usage: python3 heading_fit.py <log.csv>")
        sys.exit(1)

    path = os.path.expanduser(sys.argv[1])
    if not os.path.isfile(path):
        print(f"error: no such file: {path}")
        sys.exit(1)

    print(f"\nheading_fit.py  --  reading {path}")
    x, y, z, eul = load(path)
    n = len(x)
    if n < 50:
        print(f"error: only {n} usable rows; need a real log.")
        sys.exit(1)
    print(f"  loaded {n} rows")

    column_stats(x, y, z, eul)

    th_true, idx, disp, n_jumps = build_truth(
        x, y, LOOKAHEAD_K, MIN_DISP_M, JUMP_M)
    print("\n" + "=" * 74)
    print("2. TRUTH HEADING FROM POSITION DIFFERENCING")
    print("=" * 74)
    print(f"  centred difference, k = {LOOKAHEAD_K} samples, "
          f"min displacement = {MIN_DISP_M} m")
    print(f"  usable samples: {len(idx)} of {n - 2 * LOOKAHEAD_K} "
          f"({100.0 * len(idx) / max(n - 2 * LOOKAHEAD_K, 1):.1f}%)")
    if n_jumps:
        print(f"  WARNING: {n_jumps} position jumps > {JUMP_M} m between "
              "consecutive rows.\n           The log probably splices two "
              "recordings or contains resets.\n           Those samples are "
              "excluded, but check the log.")
    print(f"  displacement over k samples: median {np.median(disp):.3f} m, "
          f"max {disp.max():.3f} m")
    if len(idx) < 200:
        print("  WARNING: few usable samples. Lower MIN_DISP_M or raise "
              "LOOKAHEAD_K.")

    # ---- 3. Fit every candidate -------------------------------------------
    print("\n" + "=" * 74)
    print("3. CANDIDATE FIT  (theta_csv * sign  ==  theta_true + theta0)")
    print("=" * 74)

    results = []
    for ci, c in enumerate(EULER_COLS):
        th_csv = eul[idx, ci]
        for sign in (+1.0, -1.0):
            r = wrap_pi(sign * th_csv - th_true)
            off = circ_mean(r)
            sd = circ_std(wrap_pi(r - off))
            results.append({"col": c, "sign": int(sign), "ci": ci,
                            "offset": off, "std": sd})

    results.sort(key=lambda d: d["std"])

    print(f"{'rank':>4}  {'col':>4} {'sign':>5} "
          f"{'offset(rad)':>12} {'offset(deg)':>12} {'resid std':>10} "
          f"{'smear@8m':>10}")
    print("-" * 74)
    for i, d in enumerate(results):
        print(f"{i + 1:>4}  {d['col']:>4} {d['sign']:>+5} "
              f"{d['offset']:>12.4f} {math.degrees(d['offset']):>12.2f} "
              f"{d['std']:>10.4f} {800.0 * d['std']:>9.1f}cm")

    best = results[0]
    runner = results[1]

    # ---- 4. Stationarity of the winning residual --------------------------
    print("\n" + "=" * 74)
    print("4. STATIONARITY OF BEST FIT  (is the offset constant?)")
    print("=" * 74)
    r_best = wrap_pi(best["sign"] * eul[idx, best["ci"]] - th_true)
    bins = np.array_split(r_best, N_TIME_BINS)
    print(f"{'bin':>4} {'n':>7} {'offset(rad)':>12} {'offset(deg)':>12} "
          f"{'std':>9}")
    print("-" * 74)
    offs = []
    for i, b in enumerate(bins):
        if len(b) == 0:
            continue
        o = circ_mean(b)
        offs.append(o)
        print(f"{i + 1:>4} {len(b):>7} {o:>12.4f} "
              f"{math.degrees(o):>12.2f} {circ_std(wrap_pi(b - o)):>9.4f}")
    spread = circ_std(np.asarray(offs))
    print(f"\n  spread of per-bin offsets: {spread:.4f} rad "
          f"({math.degrees(spread):.2f} deg)")

    # ---- 5. Verdict --------------------------------------------------------
    print("\n" + "=" * 74)
    print("5. VERDICT")
    print("=" * 74)
    sep = runner["std"] / best["std"] if best["std"] > 1e-9 else float("inf")

    if best["std"] < 0.05:
        print(f"  IDENTIFIED. Yaw is parts[{best['col']}], "
              f"sign {best['sign']:+d}.")
        print(f"  Constant offset theta0 = {best['offset']:.4f} rad "
              f"({math.degrees(best['offset']):.2f} deg).")
        print(f"  Residual std {best['std']:.4f} rad "
              f"({math.degrees(best['std']):.2f} deg) -- tight.")
        print(f"  Beats the next distinct candidate by {sep:.1f}x.")
        print("\n  Fix in build_map.py -- use:")
        print(f"      theta = {best['sign']:+d} * parts[{best['col']}]"
              f" - ({best['offset']:.6f})")
        print("\n  Still undetermined by this test: the LiDAR beam ORDER")
        print("  (whether beam index increases clockwise or anticlockwise).")
        print("  That is one binary. Try both and keep the sharper map.")
        if spread > 0.05:
            print(f"\n  CAUTION: per-bin offsets spread {spread:.4f} rad. "
                  "The offset is\n  not perfectly constant -- expect some "
                  "residual smear.")
    elif best["std"] < 0.30:
        print(f"  PARTIAL. Best is parts[{best['col']}], "
              f"sign {best['sign']:+d}, std {best['std']:.4f} rad.")
        print(f"  {math.degrees(best['std']):.1f} deg of scatter still "
              "smears an 8 m beam by "
              f"{8.0 * best['std'] * 100:.0f} cm.")
        print("  Either the offset is not constant (see section 4) or there")
        print("  is a second error source. Check the stationarity table.")
    else:
        print(f"  NOT IDENTIFIED. Best residual std {best['std']:.4f} rad "
              "-- no candidate\n  column tracks the motion-derived heading.")
        print("  Do NOT proceed to a theta0 sweep. Possibilities:")
        print("    - yaw is not in parts[9..11] at all")
        print("    - parts[5],[6] are not the ground plane (check posZ above)")
        print("    - the log interleaves resets, so consecutive rows are not")
        print("      consecutive in time (check for position jumps)")

    print(f"\n  Smear predicted at 8 m range from residual: "
          f"{8.0 * best['std'] * 100:.1f} cm "
          f"(observed smear is 20-30 cm).")
    print()


if __name__ == "__main__":
    main()
