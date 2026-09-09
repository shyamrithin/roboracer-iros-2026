#!/usr/bin/env python3
# =============================================================================
# scan_probe.py
# -----------------------------------------------------------------------------
# Project : RoboRacer Sim Racing League @ IROS 2026 - Team CEM Navigators (26)
# Author  : Shyam Rithin
# Location: ~/roboracer/tools/scan_probe.py
# Usage   : python3 ~/roboracer/tools/scan_probe.py "~/mapdata/V1 Log.csv"
# =============================================================================
#
# DESCRIPTION
# -----------
# Follow-up to heading_fit.py, which established that the pose data is sound:
# yaw is parts[10], sign +1, offset 0.0000 rad, residual 0.57 deg. Heading is
# therefore NOT the cause of the smeared map. This script chases what is.
#
# It answers three questions, in order.
#
# SECTION 1 - SCAN INVENTORY
#   How many beams per scan, and are all scans the same length? What is the
#   actual maximum range returned? What fraction of beams are +inf, NaN, zero,
#   or sitting at max range? build_map.py treats +inf as the horizon, which
#   plants a point at horizon distance rather than discarding the beam -- if
#   those are common, they paint a filled ring around the whole trajectory.
#
# SECTION 2 - THE HALO TEST
#   The map extent (~21 x 31 m) exceeds the trajectory bounding box
#   (3.76 x 14.05 m from heading_fit.py) by ~8.6 m on the X axis and ~8.5 m on
#   the Y axis. Equal inflation on both axes is not what a projection error
#   looks like; it is what a uniform ring of max-range returns looks like.
#   This section projects the scans twice -- once with every beam, once with
#   only genuine finite returns -- and compares the two bounding boxes. If the
#   finite-only box collapses to roughly the trajectory box plus a corridor
#   half-width, the extent anomaly is fully explained and the only remaining
#   defect is wall THICKNESS.
#
# SECTION 3 - OFFSET AND BEAM-ORDER SWEEP
#   Two unknowns remain. (a) The LiDAR sits forward of the vehicle origin;
#   the Technical Guide gives 0.2733 m. If build_map.py projects from the
#   pose origin instead of the sensor, every point displaces by that amount
#   in a body-fixed direction that ROTATES with the car -- which smears walls
#   by up to 0.27 m, matching the observed 20-30 cm almost exactly. (b) Beam
#   order: whether index 0 is at -135 deg or +135 deg. heading_fit.py could
#   not resolve this one.
#
#   Both are swept together and scored by SHARPNESS, defined as
#
#       sum(count_i^2) / (sum(count_i))^2
#
#   over occupied cells -- the normalised second moment of the hit histogram.
#   This rewards CONCENTRATING returns into few cells. The earlier
#   convention.py scored by occupied-cell count, which is why its best option
#   led by only 12%: cell count is nearly flat across conventions because a
#   wrong convention scatters points without changing how many there are.
#   Sharpness separates by factors. A correct convention should show a clear
#   single peak in the offset sweep near the true sensor offset.
#
# INTERPRETING SECTION 3
#   - Peak at ~0.27 m with a clear ridge  -> missing sensor offset is the bug.
#   - Peak at ~0.00 m                     -> build_map already handles it;
#                                            smear is from something else.
#   - Flat / no peak                      -> the projection model is wrong at
#                                            a deeper level. STOP, do not
#                                            tune. Send me the output.
#
# Read-only. Writes nothing, modifies no map code.
# =============================================================================

import sys
import os
import math
import numpy as np

# ---- CSV layout ------------------------------------------------------------
COL_POS_X = 5
COL_POS_Y = 6
COL_YAW = 10          # confirmed by heading_fit.py: sign +1, offset ~0
COL_LIDAR = 21

# ---- LiDAR geometry (Technical Guide) --------------------------------------
FOV_TOTAL_DEG = 270.0
LIDAR_FWD_NOMINAL_M = 0.2733   # sensor forward of vehicle origin

# ---- Analysis settings -----------------------------------------------------
GRID_RES_M = 0.05
SUBSAMPLE = 4                  # use every Nth scan for the sweep
MAX_SCANS_PARSE = 4500
OFFSET_SWEEP = np.arange(-0.10, 0.51, 0.025)
NEAR_MAX_FRAC = 0.98           # beams within this fraction of max = "at max"


def parse(path, subsample):
    """Parse poses and scans. Returns (x, y, yaw, scans, beam_len_set)."""
    xs, ys, yaws, scans = [], [], [], []
    lens = {}
    n_rows = 0
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            n_rows += 1
            if (n_rows - 1) % subsample:
                continue
            if n_rows > MAX_SCANS_PARSE * subsample:
                break
            parts = line.split(",")
            if len(parts) <= COL_LIDAR:
                continue
            try:
                px = float(parts[COL_POS_X])
                py = float(parts[COL_POS_Y])
                th = float(parts[COL_YAW])
            except ValueError:
                continue
            toks = parts[COL_LIDAR].split()
            if not toks:
                continue
            r = np.array([float(t) if t not in ("inf", "Infinity")
                          else np.inf for t in toks], dtype=np.float64)
            lens[len(r)] = lens.get(len(r), 0) + 1
            xs.append(px)
            ys.append(py)
            yaws.append(th)
            scans.append(r)
    return (np.asarray(xs), np.asarray(ys), np.asarray(yaws), scans,
            lens, n_rows)


def inventory(scans, lens, n_rows, n_used):
    print("\n" + "=" * 74)
    print("1. SCAN INVENTORY")
    print("=" * 74)
    print(f"  rows in file: {n_rows}   scans parsed: {n_used} "
          f"(every {SUBSAMPLE}th)")
    print("\n  beams per scan:")
    for L, c in sorted(lens.items()):
        print(f"    {L} beams  x {c} scans")
    if len(lens) > 1:
        print("    WARNING: beam count is NOT constant. The angular increment")
        print("    must then be computed per scan, not once.")

    allr = np.concatenate(scans)
    n = allr.size
    n_inf = int(np.sum(np.isinf(allr)))
    n_nan = int(np.sum(np.isnan(allr)))
    finite = allr[np.isfinite(allr)]
    n_zero = int(np.sum(finite <= 1e-6))
    pos = finite[finite > 1e-6]
    rmax = float(pos.max()) if pos.size else float("nan")
    n_atmax = int(np.sum(pos >= NEAR_MAX_FRAC * rmax))

    print(f"\n  total beams:        {n}")
    print(f"  +inf:               {n_inf:>10}  ({100.0*n_inf/n:5.2f}%)")
    print(f"  NaN:                {n_nan:>10}  ({100.0*n_nan/n:5.2f}%)")
    print(f"  zero/near-zero:     {n_zero:>10}  ({100.0*n_zero/n:5.2f}%)")
    print(f"  at >= {NEAR_MAX_FRAC:.0%} of max:  {n_atmax:>10}  "
          f"({100.0*n_atmax/n:5.2f}%)")
    print(f"\n  max finite range:   {rmax:.4f} m")
    if pos.size:
        qs = np.percentile(pos, [1, 25, 50, 75, 95, 99])
        print(f"  percentiles (m): 1%={qs[0]:.2f}  25%={qs[1]:.2f}  "
              f"50%={qs[2]:.2f}  75%={qs[3]:.2f}  95%={qs[4]:.2f}  "
              f"99%={qs[5]:.2f}")

    print("\n  range histogram:")
    hist, edges = np.histogram(pos, bins=20)
    peak = hist.max() if hist.size else 1
    for i in range(len(hist)):
        bar = "#" * int(48.0 * hist[i] / peak)
        print(f"    {edges[i]:5.2f}-{edges[i+1]:5.2f} m |{bar} {hist[i]}")

    far_frac = (n_inf + n_atmax) / n
    print(f"\n  beams that would be planted at the horizon: "
          f"{100.0*far_frac:.2f}%")
    if far_frac > 0.02:
        print("  -> Enough to paint a visible ring. These are the 'haze of")
        print("     stray points'. They must be DISCARDED, not clipped.")
    else:
        print("  -> Too few to explain a large halo on their own.")
    return rmax


def project(x, y, yaw, scans, fwd_off, beam_sign, rmax,
            drop_far=True, res=GRID_RES_M):
    """
    Project scans to world points.
      fwd_off   : sensor position forward of the pose origin (m)
      beam_sign : +1 or -1, direction of increasing beam index
      drop_far  : discard inf / at-max beams instead of planting them
    Returns (px, py) world coordinates.
    """
    half = math.radians(FOV_TOTAL_DEG) / 2.0
    out_x, out_y = [], []
    for i in range(len(scans)):
        r = scans[i]
        nb = r.size
        if nb < 2:
            continue
        inc = math.radians(FOV_TOTAL_DEG) / (nb - 1)   # 1081 beams -> n-1
        ang = -half + beam_sign * inc * np.arange(nb)

        ok = np.isfinite(r) & (r > 1e-6)
        if drop_far:
            ok &= (r < NEAR_MAX_FRAC * rmax)
        if not np.any(ok):
            continue
        rr = r[ok]
        aa = ang[ok]

        th = yaw[i]
        # sensor origin in world
        sx = x[i] + fwd_off * math.cos(th)
        sy = y[i] + fwd_off * math.sin(th)
        wa = th + aa
        out_x.append(sx + rr * np.cos(wa))
        out_y.append(sy + rr * np.sin(wa))

    if not out_x:
        return np.array([]), np.array([])
    return np.concatenate(out_x), np.concatenate(out_y)


def sharpness(px, py, res=GRID_RES_M):
    """
    Normalised second moment of the hit histogram:  sum(c^2) / (sum c)^2.
    Higher = returns concentrated into fewer cells = sharper walls.
    Scale-free, so it is comparable across conventions.
    Also returns occupied cell count and bounding box for reference.
    """
    if px.size == 0:
        return 0.0, 0, (0.0, 0.0)
    ix = np.floor(px / res).astype(np.int64)
    iy = np.floor(py / res).astype(np.int64)
    ix -= ix.min()
    iy -= iy.min()
    w = int(ix.max()) + 1
    key = iy.astype(np.int64) * w + ix
    counts = np.bincount(key)
    counts = counts[counts > 0]
    tot = counts.sum()
    s = float(np.sum(counts.astype(np.float64) ** 2)) / float(tot) ** 2
    span = (float(px.max() - px.min()), float(py.max() - py.min()))
    return s, int(counts.size), span


def halo_test(x, y, yaw, scans, rmax):
    print("\n" + "=" * 74)
    print("2. HALO TEST  (does discarding far beams fix the extent?)")
    print("=" * 74)
    tx = x.max() - x.min()
    ty = y.max() - y.min()
    print(f"  trajectory bounding box:      {tx:6.2f} x {ty:6.2f} m")

    for label, drop in (("ALL beams (inf clipped to horizon)", False),
                        ("FINITE beams only (far discarded)", True)):
        px, py = project(x, y, yaw, scans, 0.0, +1, rmax, drop_far=drop)
        if px.size == 0:
            print(f"  {label:<36}  no points")
            continue
        s, cells, span = sharpness(px, py)
        print(f"  {label:<36}  {span[0]:6.2f} x {span[1]:6.2f} m   "
              f"({px.size} pts, {cells} cells)")

    print("\n  Reference: the reported map extent was ~21 x 31 m.")
    print("  If FINITE-only collapses to roughly the trajectory box plus a")
    print("  corridor half-width on each side, the extent anomaly is fully")
    print("  explained and only wall THICKNESS remains.")


def sweep(x, y, yaw, scans, rmax):
    print("\n" + "=" * 74)
    print("3. OFFSET AND BEAM-ORDER SWEEP  (scored by sharpness)")
    print("=" * 74)
    print(f"  metric: sum(c^2)/(sum c)^2 over occupied cells, "
          f"{GRID_RES_M*100:.0f} cm grid")
    print(f"  higher = sharper. far beams discarded.\n")

    best = None
    results = {}
    for bs in (+1, -1):
        vals = []
        for off in OFFSET_SWEEP:
            px, py = project(x, y, yaw, scans, float(off), bs, rmax,
                             drop_far=True)
            s, cells, span = sharpness(px, py)
            vals.append(s)
            if best is None or s > best[0]:
                best = (s, float(off), bs, cells, span)
        results[bs] = np.asarray(vals)

    peak = max(results[+1].max(), results[-1].max())
    print(f"  {'offset(m)':>10} {'beam +1':>12} {'beam -1':>12}   "
          f"{'profile (both)':<30}")
    print("-" * 74)
    for i, off in enumerate(OFFSET_SWEEP):
        a, b = results[+1][i], results[-1][i]
        bar_a = "#" * int(22.0 * a / peak)
        bar_b = "." * int(22.0 * b / peak)
        mark = " <<<" if max(a, b) >= peak - 1e-15 else ""
        print(f"  {off:>10.3f} {a:>12.3e} {b:>12.3e}   "
              f"{bar_a:<24}{mark}")

    s, off, bs, cells, span = best
    print("\n" + "-" * 74)
    print(f"  BEST: offset {off:.3f} m, beam order {bs:+d}, "
          f"sharpness {s:.4e}")
    print(f"        {cells} occupied cells, extent "
          f"{span[0]:.2f} x {span[1]:.2f} m")

    # separation: best vs offset 0 with same beam order
    i0 = int(np.argmin(np.abs(OFFSET_SWEEP - 0.0)))
    s0 = results[bs][i0]
    ratio = s / s0 if s0 > 0 else float("inf")
    print(f"        vs offset 0.000 (same beam order): {ratio:.2f}x sharper")

    print("\n" + "=" * 74)
    print("4. VERDICT")
    print("=" * 74)
    if abs(off - LIDAR_FWD_NOMINAL_M) < 0.06 and ratio > 1.15:
        print(f"  Peak at {off:.3f} m matches the Technical Guide value "
              f"{LIDAR_FWD_NOMINAL_M} m.")
        print("  The missing sensor offset is the smear. Apply it in")
        print("  build_map.py: project from the SENSOR origin, not the pose")
        print("  origin:")
        print(f"      sx = posX + {off:.4f} * cos(yaw)")
        print(f"      sy = posY + {off:.4f} * sin(yaw)")
        print(f"  Use beam order {bs:+d}.")
    elif abs(off) < 0.06 and ratio < 1.15:
        print("  Peak sits at ~0 m: build_map.py is already projecting from")
        print("  the right origin. The sensor offset is NOT the smear.")
        print("  Remaining suspects: heading noise (7.9 cm at 8 m), grid")
        print("  resolution floor, or a per-beam angular error.")
    elif ratio < 1.05:
        print("  FLAT. No offset materially sharpens the map, which means")
        print("  the projection model is wrong at a deeper level than a")
        print("  translation. Do NOT tune. Send this output back.")
    else:
        print(f"  Peak at {off:.3f} m, {ratio:.2f}x sharper than zero, but it")
        print(f"  does not match the nominal {LIDAR_FWD_NOMINAL_M} m.")
        print("  Treat as provisional: apply it, rebuild, and inspect the")
        print("  map visually before trusting it.")
    print()


def main():
    if len(sys.argv) < 2:
        print("usage: python3 scan_probe.py <log.csv>")
        sys.exit(1)
    path = os.path.expanduser(sys.argv[1])
    if not os.path.isfile(path):
        print(f"error: no such file: {path}")
        sys.exit(1)

    print(f"\nscan_probe.py  --  reading {path}")
    print("  (parsing LiDAR fields, this takes a few seconds)")
    x, y, yaw, scans, lens, n_rows = parse(path, SUBSAMPLE)
    if len(scans) < 20:
        print(f"error: only {len(scans)} usable scans.")
        sys.exit(1)

    rmax = inventory(scans, lens, n_rows, len(scans))
    halo_test(x, y, yaw, scans, rmax)
    sweep(x, y, yaw, scans, rmax)


if __name__ == "__main__":
    main()
