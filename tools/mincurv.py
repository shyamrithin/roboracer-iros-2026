#!/usr/bin/env python3
# =============================================================================
# mincurv.py
# RoboRacer IROS 2026 / CEM Navigators
# =============================================================================
#
# CODE DESCRIPTION
# -----------------------------------------------------------------------------
# Minimum-curvature raceline optimiser. Given a reference path and an
# occupancy map, it finds a lateral offset at every waypoint that minimises
# integrated squared curvature, subject to staying inside the corridor. It
# then re-derives the velocity profile from the new geometry.
#
# WHY THIS AND NOT THE TUM OPTIMISER
#   The TUM global optimiser rejects this track outright: its spline normals
#   cross at the two hairpins, where the medial-axis centreline turns 26 deg
#   per 0.11 m - a 0.24 m radius against the vehicle's 0.56 m minimum. Seven
#   configurations were tried and all failed for the same reason. TUM assumes
#   corners that are large relative to the vehicle; these are not.
#
#   This solver sidesteps that by never building spline normals. It works
#   directly on the offset along a fixed normal field, so a tight corner is
#   simply a region where the feasible offset range is narrow.
#
# WHY A NEW LINE SHOULD BE FASTER
#   Every raceline in this project so far is the reactive gap follower's
#   driven path, smoothed. The follower aims at the deepest gap, which in a
#   corner is the inside, so the recorded line turns in early and apexes
#   early, leaving its radius near the vehicle's limit. Logged apex samples
#   showed L=3.7 m against R=0.5 m - nearly four metres of unused track on the
#   outside of the turn. The corridor is 1.73 m at its narrowest and 1.90 m
#   median, so the room for a wider entry genuinely exists.
#
#   Because v = sqrt(a_lat * R), a larger minimum radius directly permits more
#   speed at the same grip limit. Minimum curvature is the standard objective
#   for exactly this reason.
#
# METHOD
#   Let the reference path be p(s) with unit normal n(s). The optimised line
#   is q(s) = p(s) + a(s) * n(s) for a scalar offset a(s). Curvature of q is
#   approximated, for small offsets, by the second derivative of a along the
#   path plus the reference curvature. Minimising the integral of that squared
#   is then a quadratic program in a:
#
#       minimise    || D2 a + k_ref ||^2  +  lambda * || D1 a ||^2
#       subject to  a_min(s) <= a(s) <= a_max(s)
#
#   D1 and D2 are periodic finite-difference operators, so the line closes.
#   The bounds come from ray casting against the occupancy map, inset by the
#   vehicle half-width plus a safety margin. The problem is solved by
#   projected gradient descent, which handles the box constraints directly and
#   needs no external solver - scipy's optimisers are not available for a
#   problem of this size at this resolution.
#
#   The regularisation term penalises rapid changes in offset, which keeps the
#   line smooth where the corridor is wide and the curvature term is weak.
#
# USAGE
#   python3 mincurv.py ~/mapdata/raceline_v7_r.csv ~/mapdata/bridge_ips_cl.yaml \
#       -o ~/mapdata/raceline_mc.csv
#
#   --margin controls how much clearance is kept from the walls. Start at 0.30
#   and reduce only if the result is clean. --lambda-smooth trades smoothness
#   against curvature; raise it if the line looks nervous.
#
# DEPENDENCIES: numpy, Pillow
# =============================================================================

import argparse
import sys

import numpy as np
from PIL import Image


VEHICLE_HALF_WIDTH = 0.138          # 0.2762 m wide


def load_map(path):
    cfg = {}
    for line in open(path):
        line = line.strip()
        if not line or line.startswith('#') or ':' not in line:
            continue
        k, v = line.split(':', 1)
        cfg[k.strip()] = v.strip()
    base = path.rsplit('/', 1)[0] if '/' in path else '.'
    grid = np.flipud(np.array(Image.open(f"{base}/{cfg['image']}")))
    res = float(cfg['resolution'])
    org = [float(x) for x in cfg['origin'].strip('[]').split(',')[:2]]
    return (grid == 254), res, org


def cast(free, res, org, x, y, dx, dy, limit):
    """Distance to the first non-free cell along (dx, dy), in metres."""
    H, W = free.shape
    step = res * 0.5
    d = 0.0
    while d < limit:
        d += step
        cx = int((x + dx * d - org[0]) / res)
        cy = int((y + dy * d - org[1]) / res)
        if cx < 0 or cx >= W or cy < 0 or cy >= H or not free[cy, cx]:
            return d - step
    return limit


def periodic_diff_matrices(n, ds):
    """First and second derivative operators on a closed path."""
    i = np.arange(n)
    D1 = np.zeros((n, n))
    D1[i, (i + 1) % n] = 0.5 / ds
    D1[i, (i - 1) % n] = -0.5 / ds
    D2 = np.zeros((n, n))
    D2[i, i] = -2.0 / (ds * ds)
    D2[i, (i + 1) % n] = 1.0 / (ds * ds)
    D2[i, (i - 1) % n] = 1.0 / (ds * ds)
    return D1, D2


def curvature(x, y):
    dx, dy = np.gradient(x), np.gradient(y)
    ddx, ddy = np.gradient(dx), np.gradient(dy)
    den = (dx * dx + dy * dy) ** 1.5
    den = np.where(den < 1e-9, 1e-9, den)
    return (dx * ddy - dy * ddx) / den


def smooth_periodic(v, k):
    if k < 2:
        return v
    n = len(v)
    return np.convolve(np.r_[v, v, v], np.ones(k) / k, 'same')[n:2 * n]


def velocity_profile(x, y, kappa, a_lat, a_acc, a_dec, v_max, v_min):
    n = len(x)
    seg = np.hypot(np.diff(np.append(x, x[0])), np.diff(np.append(y, y[0])))
    v = np.sqrt(a_lat / np.maximum(np.abs(kappa), 1e-6))
    v = np.clip(v, v_min, v_max)
    for _ in range(3):
        for i in range(n):
            j = (i + 1) % n
            v[j] = min(v[j], np.sqrt(v[i] ** 2 + 2 * a_acc * seg[i]))
        for i in range(n - 1, -1, -1):
            j = (i + 1) % n
            v[i] = min(v[i], np.sqrt(v[j] ** 2 + 2 * a_dec * seg[i]))
    return v, seg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('raceline')
    ap.add_argument('map_yaml')
    ap.add_argument('-o', '--out', default='raceline_mc.csv')
    ap.add_argument('--margin', type=float, default=0.30,
                    help='clearance kept from the wall, beyond half-width')
    ap.add_argument('--lambda-smooth', type=float, default=15.0)
    ap.add_argument('--iters', type=int, default=4000)
    ap.add_argument('--step', type=float, default=2e-4)
    ap.add_argument('--a-lat', type=float, default=4.8)
    ap.add_argument('--a-acc', type=float, default=3.0)
    ap.add_argument('--a-dec', type=float, default=2.2)
    ap.add_argument('--v-max', type=float, default=6.5)
    ap.add_argument('--v-min', type=float, default=1.5)
    args = ap.parse_args()

    rows = [l for l in open(args.raceline) if l.strip()]
    hdr = rows[0]
    P = np.array([[float(v) for v in l.split(',')] for l in rows[1:]])
    x0, y0 = P[:, 0].copy(), P[:, 1].copy()
    n = len(x0)

    free, res, org = load_map(args.map_yaml)

    seg = np.hypot(np.diff(np.append(x0, x0[0])),
                   np.diff(np.append(y0, y0[0])))
    ds = float(np.mean(seg))
    dx, dy = np.gradient(x0), np.gradient(y0)
    h = np.hypot(dx, dy)
    nx, ny = -dy / h, dx / h                     # left normal

    # ---- feasible offset range at every waypoint --------------------------
    inset = VEHICLE_HALF_WIDTH + args.margin
    a_max = np.empty(n)
    a_min = np.empty(n)
    for i in range(n):
        left = cast(free, res, org, x0[i], y0[i], nx[i], ny[i], 3.0)
        right = cast(free, res, org, x0[i], y0[i], -nx[i], -ny[i], 3.0)
        a_max[i] = max(left - inset, 0.0)
        a_min[i] = -max(right - inset, 0.0)
    width = a_max - a_min
    print(f'  {n} waypoints, spacing {ds*100:.1f} cm')
    print(f'  usable lateral range: min {width.min():.2f} '
          f'median {np.median(width):.2f} max {width.max():.2f} m')

    k_ref = smooth_periodic(curvature(x0, y0), 7)
    R_ref = 1.0 / np.maximum(np.abs(k_ref), 1e-6)
    print(f'  reference radius: min {R_ref.min():.2f} '
          f'median {np.median(R_ref):.2f} m')

    D1, D2 = periodic_diff_matrices(n, ds)

    # ---- projected gradient descent ---------------------------------------
    # objective  f(a) = ||D2 a + k_ref||^2 + lam ||D1 a||^2
    # gradient   2 D2^T (D2 a + k_ref) + 2 lam D1^T D1 a
    lam = args.lambda_smooth
    A = 2.0 * (D2.T @ D2 + lam * (D1.T @ D1))
    b = 2.0 * (D2.T @ k_ref)
    a = np.zeros(n)
    step = args.step
    prev = None
    for it in range(args.iters):
        g = A @ a + b
        a_new = np.clip(a - step * g, a_min, a_max)
        if prev is not None and np.max(np.abs(a_new - a)) < 1e-7:
            print(f'  converged at iteration {it}')
            a = a_new
            break
        prev = a
        a = a_new
    else:
        print(f'  ran {args.iters} iterations')

    print(f'  offset: min {a.min():+.3f} max {a.max():+.3f} '
          f'mean |a| {np.mean(np.abs(a)):.3f} m')

    xn = x0 + a * nx
    yn = y0 + a * ny

    k_new = smooth_periodic(curvature(xn, yn), 7)
    R_new = 1.0 / np.maximum(np.abs(k_new), 1e-6)
    print(f'  optimised radius: min {R_new.min():.2f} '
          f'median {np.median(R_new):.2f} m')

    # the number that matters: does the tightest corner open up?
    worst_before = np.sort(R_ref)[:20].mean()
    worst_after = np.sort(R_new)[:20].mean()
    print(f'  mean radius of the 20 tightest points: '
          f'{worst_before:.2f} -> {worst_after:.2f} m')

    v_old, seg_old = velocity_profile(x0, y0, k_ref, args.a_lat, args.a_acc,
                                      args.a_dec, args.v_max, args.v_min)
    v_new, seg_new = velocity_profile(xn, yn, k_new, args.a_lat, args.a_acc,
                                      args.a_dec, args.v_max, args.v_min)
    t_old = float(np.sum(seg_old / np.maximum(v_old, 0.1)))
    t_new = float(np.sum(seg_new / np.maximum(v_new, 0.1)))
    print(f'  path length {seg_old.sum():.2f} -> {seg_new.sum():.2f} m')
    print(f'  predicted lap {t_old:.2f} -> {t_new:.2f} s ({t_new-t_old:+.2f})')
    print(f'  speed min {v_new.min():.2f} mean {v_new.mean():.2f} '
          f'max {v_new.max():.2f} m/s')

    out = np.column_stack([xn, yn, v_new, k_new,
                           np.cumsum(seg_new) - seg_new[0]])
    with open(args.out, 'w') as f:
        f.write(hdr)
        for r in out:
            f.write(','.join('%.4f' % q for q in r) + '\n')
    print(f'  wrote {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
