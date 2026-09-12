#!/usr/bin/env python3
# =============================================================================
# make_raceline.py
# -----------------------------------------------------------------------------
# Project : RoboRacer Sim Racing League @ IROS 2026 - Team CEM Navigators (26)
# Author  : Shyam Rithin
# Location: ~/roboracer/tools/make_raceline.py
# Usage   : python3 make_raceline.py "~/mapdata/V1 Log.csv" -o ~/mapdata/raceline.csv
# =============================================================================
#
# DESCRIPTION
# -----------
# Builds a racing line and velocity profile from the recorded data-recorder log,
# and writes it as waypoints in the MAP frame for a path-tracking controller.
#
# The recorded trajectory is a driven line, not an optimal one - it is what the
# reactive gap follower produced. Two things are done to it.
#
#   1. GEOMETRY. Laps are detected, one representative lap is selected, and it
#      is resampled to uniform arc length and smoothed with a periodic spline.
#      Smoothing alone is worth something: the reactive line contains
#      scan-to-scan jitter that a tracker would otherwise chase.
#
#   2. VELOCITY. This is where the lap time actually is. The reactive car runs
#      near-constant speed because throttle is a constant cut by steering
#      angle. A stored line has known curvature everywhere ahead, so speed can
#      be planned: fast where the line is straight, slow where it is not.
#
# COORDINATE FRAME
# ----------------
# The log is in world coordinates. The tracker consumes map -> base_link from
# slam_toolbox, so the waypoints must be in the MAP frame. Measured with the
# vehicle stationary at the start pose, map -> odom was
#
#     translation (-0.262, -0.002), yaw -0.042 rad
#
# and odom coincides with world at that instant because dead reckoning is
# seeded with the true start pose. So
#
#     p_map = R(-0.042) * p_world + (-0.262, -0.002)
#
# These are parameters, not constants, because they are specific to the map
# that was built. Re-measure them after building a map on a new track: run the
# stack with the vehicle stationary and read `ros2 run tf2_ros tf2_echo map odom`.
#
# VELOCITY PROFILE
# ----------------
# Three limits, applied in sequence:
#
#   * Lateral grip.   v = sqrt(a_lat_max / |kappa|), capped at v_max.
#   * Acceleration.   A forward pass limits how fast speed may rise:
#                     v[i+1] <= sqrt(v[i]^2 + 2 * a_acc_max * ds).
#   * Deceleration.   A backward pass limits how fast it may fall:
#                     v[i] <= sqrt(v[i+1]^2 + 2 * a_dec_max * ds).
#
# Both passes are run twice because the line is a closed loop and the first
# pass has no valid starting value.
#
# a_dec_max MATTERS MOST AND IS THE LEAST KNOWN. This vehicle has no brake:
# negative throttle engages reverse, so deceleration comes from idle torque
# and tyre scrub alone. The default here is deliberately conservative. It
# should be measured on the vehicle before the profile is trusted - command
# zero throttle at a known speed and time the decay.
#
# OUTPUT
# ------
# CSV with a header: x_map, y_map, v_target, kappa, s. One row per waypoint.
# =============================================================================

import argparse
import math
import os
import sys

import numpy as np

# ---- CSV layout (AutoDRIVE data recorder, per the Technical Guide) ----------
COL_POS_X = 5
COL_POS_Y = 6
COL_YAW = 10


def load_track(path):
    """Load world-frame positions from the data-recorder log."""
    xs, ys = [], []
    bad = 0
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) <= COL_YAW:
                bad += 1
                continue
            try:
                xs.append(float(parts[COL_POS_X]))
                ys.append(float(parts[COL_POS_Y]))
            except ValueError:
                bad += 1
    if bad:
        print(f"  skipped {bad} unparseable rows")
    return np.asarray(xs), np.asarray(ys)


def to_map_frame(x, y, tx, ty, yaw):
    """Rotate then translate world coordinates into the map frame."""
    c, s = math.cos(yaw), math.sin(yaw)
    return c * x - s * y + tx, s * x + c * y + ty


def split_laps(x, y, radius):
    """
    Split the trajectory into laps by detecting returns to the starting point.

    A lap ends when the path comes back within `radius` of where it began,
    having previously left that neighbourhood. Returns a list of (start, end)
    index pairs.
    """
    x0, y0 = x[0], y[0]
    d = np.hypot(x - x0, y - y0)
    inside = d < radius

    laps = []
    start = 0
    was_outside = False
    for i in range(1, len(d)):
        if not inside[i]:
            was_outside = True
        elif was_outside and inside[i] and not inside[i - 1]:
            laps.append((start, i))
            start = i
            was_outside = False
    return laps


def resample(x, y, n):
    """Resample a closed path to n points of uniform arc length."""
    xc = np.append(x, x[0])
    yc = np.append(y, y[0])
    seg = np.hypot(np.diff(xc), np.diff(yc))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    su = np.linspace(0.0, total, n, endpoint=False)
    return np.interp(su, s, xc), np.interp(su, s, yc), total


def smooth_closed(x, y, window):
    """
    Circular moving average. Simple, and adequate here: the input is a driven
    line whose noise is scan-to-scan jitter rather than structured error.
    """
    if window < 3:
        return x.copy(), y.copy()
    if window % 2 == 0:
        window += 1
    k = np.ones(window) / window
    xp = np.concatenate([x[-window:], x, x[:window]])
    yp = np.concatenate([y[-window:], y, y[:window]])
    return (np.convolve(xp, k, mode="same")[window:-window],
            np.convolve(yp, k, mode="same")[window:-window])


def curvature_closed(x, y):
    """Menger curvature from each consecutive triple of points."""
    n = len(x)
    k = np.zeros(n)
    for i in range(n):
        ax, ay = x[i - 1], y[i - 1]
        bx, by = x[i], y[i]
        cx, cy = x[(i + 1) % n], y[(i + 1) % n]
        ab = math.hypot(bx - ax, by - ay)
        bc = math.hypot(cx - bx, cy - by)
        ca = math.hypot(ax - cx, ay - cy)
        area2 = abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))
        denom = ab * bc * ca
        k[i] = 0.0 if denom < 1e-12 else 2.0 * area2 / denom
    return k


def velocity_profile(kappa, ds, a_lat, a_acc, a_dec, v_max, v_min):
    """Grip limit, then forward and backward passes for the drivetrain."""
    v = np.where(kappa > 1e-6, np.sqrt(a_lat / np.maximum(kappa, 1e-6)), v_max)
    v = np.clip(v, v_min, v_max)
    n = len(v)

    for _ in range(2):
        for i in range(n):
            j = (i + 1) % n
            v[j] = min(v[j], math.sqrt(v[i] ** 2 + 2.0 * a_acc * ds))
    for _ in range(2):
        for i in range(n - 1, -1, -1):
            j = (i - 1) % n
            v[j] = min(v[j], math.sqrt(v[i] ** 2 + 2.0 * a_dec * ds))

    return np.clip(v, v_min, v_max)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("-o", "--out", default="raceline.csv")
    ap.add_argument("--n", type=int, default=400, help="waypoint count")
    ap.add_argument("--smooth", type=int, default=15, help="smoothing window")
    ap.add_argument("--lap-radius", type=float, default=1.0)
    ap.add_argument("--map-tx", type=float, default=-0.262)
    ap.add_argument("--map-ty", type=float, default=-0.002)
    ap.add_argument("--map-yaw", type=float, default=-0.042)
    ap.add_argument("--a-lat", type=float, default=4.0, help="m/s^2")
    ap.add_argument("--a-acc", type=float, default=3.0, help="m/s^2")
    ap.add_argument("--a-dec", type=float, default=1.5,
                    help="m/s^2 - NO BRAKE on this vehicle, measure it")
    ap.add_argument("--v-max", type=float, default=8.0, help="m/s")
    ap.add_argument("--v-min", type=float, default=1.5, help="m/s")
    args = ap.parse_args()

    path = os.path.expanduser(args.log)
    if not os.path.isfile(path):
        print(f"error: no such file: {path}")
        sys.exit(1)

    print(f"\nmake_raceline.py  --  reading {path}")
    xw, yw = load_track(path)
    print(f"  {len(xw)} samples")
    print(f"  world extent  x [{xw.min():.2f},{xw.max():.2f}]  "
          f"y [{yw.min():.2f},{yw.max():.2f}]")

    x, y = to_map_frame(xw, yw, args.map_tx, args.map_ty, args.map_yaw)
    print(f"  map extent    x [{x.min():.2f},{x.max():.2f}]  "
          f"y [{y.min():.2f},{y.max():.2f}]")

    laps = split_laps(x, y, args.lap_radius)
    print(f"\nlaps detected: {len(laps)}")
    if not laps:
        print("  none found. Raise --lap-radius, or check the log covers "
              "more than one lap.")
        sys.exit(1)

    lengths = []
    for a, b in laps:
        seg = np.hypot(np.diff(x[a:b]), np.diff(y[a:b])).sum()
        lengths.append(seg)
    lengths = np.asarray(lengths)
    print(f"  lap length: mean {lengths.mean():.2f} m  "
          f"std {lengths.std():.2f}  "
          f"range [{lengths.min():.2f},{lengths.max():.2f}]")

    # The median-length lap is the representative one: shortest laps tend to
    # cut a corner the vehicle only just survived, longest tend to be the
    # recovery lap after one.
    pick = int(np.argsort(lengths)[len(lengths) // 2])
    a, b = laps[pick]
    print(f"  using lap {pick + 1} ({lengths[pick]:.2f} m, "
          f"{b - a} samples)")

    lx, ly = x[a:b], y[a:b]
    rx, ry, total = resample(lx, ly, args.n)
    sx, sy = smooth_closed(rx, ry, args.smooth)
    sx, sy, total = resample(sx, sy, args.n)
    ds = total / args.n

    kappa = curvature_closed(sx, sy)
    v = velocity_profile(kappa, ds, args.a_lat, args.a_acc,
                         args.a_dec, args.v_max, args.v_min)

    print(f"\nraceline: {args.n} waypoints, {total:.2f} m, "
          f"spacing {ds * 100:.1f} cm")
    r = np.where(kappa > 1e-6, 1.0 / np.maximum(kappa, 1e-6), np.inf)
    print(f"  radius: min {r.min():.2f} m  median {np.median(r):.2f} m")
    print(f"  speed : min {v.min():.2f}  mean {v.mean():.2f}  "
          f"max {v.max():.2f} m/s")

    lap_time = float(np.sum(ds / np.maximum(v, 1e-3)))
    print(f"\n  predicted lap time {lap_time:.2f} s")
    print(f"  reactive stack measures 7.5-7.6 s at a near-constant "
          f"{total / 7.55:.2f} m/s")
    if lap_time > 7.5:
        print("  -> SLOWER than the reactive stack. Raise a_lat or a_dec, or")
        print("     accept that this line is not worth tracking.")

    print("\n  speed profile (one marker per 5 per cent of the lap):")
    step = max(args.n // 20, 1)
    for i in range(0, args.n, step):
        bar = "#" * int(40 * v[i] / max(v.max(), 1e-6))
        print(f"    s={i * ds:6.2f} m  v={v[i]:5.2f}  R={min(r[i], 99):5.1f}  {bar}")

    out = os.path.expanduser(args.out)
    s_arr = np.arange(args.n) * ds
    with open(out, "w") as fh:
        fh.write("x_map,y_map,v_target,kappa,s\n")
        for i in range(args.n):
            fh.write(f"{sx[i]:.4f},{sy[i]:.4f},{v[i]:.4f},"
                     f"{kappa[i]:.6f},{s_arr[i]:.4f}\n")
    print(f"\nwrote {out}")
    print("\nBefore trusting this, overlay it on the map in rviz and confirm "
          "it lies\ninside the corridor. If it does not, the map-frame "
          "transform is wrong.")


if __name__ == "__main__":
    main()
