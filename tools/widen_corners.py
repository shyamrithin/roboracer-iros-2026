#!/usr/bin/env python3
# =============================================================================
# widen_corners.py
# RoboRacer IROS 2026 / CEM Navigators
# =============================================================================
#
# CODE DESCRIPTION
# -----------------------------------------------------------------------------
# Widens a raceline through its medium-radius corners, then re-derives the
# velocity profile from the new curvature. The aim is a higher minimum radius,
# which lets the same lateral grip limit carry more speed.
#
# WHY
#   Every raceline in this project is the reactive gap follower's driven path,
#   smoothed. That path hugs the inside of every corner - the follower aims at
#   the deepest gap, which in a corner is the inside - so its radius stays near
#   the vehicle's limit and the velocity profile is capped accordingly. Logged
#   apex samples showed L=3.7 m against R=0.5 m: nearly four metres of unused
#   track on the outside of the turn.
#
#   Raising a_lat does not help, because the radius is the binding constraint:
#   at a_lat 5.5 the vehicle ran wide and collided, while 4.8 is clean. Raising
#   v_max does not help either, because the straight is limited by the
#   vehicle's acceleration rather than by the target - measured v exceeded the
#   target by a similar margin whether the cap was 6.50 or 7.25 m/s.
#
#   Geometry is the remaining lever. A classic out-in-out line through a corner
#   raises the effective radius, and v = sqrt(a_lat * R) then permits more
#   speed at the same grip limit.
#
# WHAT IS DELIBERATELY NOT TOUCHED
#   The two hairpins. Their radius is about 0.5 m against the vehicle's 0.56 m
#   minimum turning radius, so there is no room to widen and nothing to gain -
#   this is the same property that made the TUM global optimiser reject the
#   track outright. Only corners above --min-radius are modified.
#
#   The upper span between the towers (pose 1) is also excluded by default.
#   Every configuration that moved the line there made things worse: a 5 cm
#   shift toward the inside produced 2 collisions in 18 laps.
#
# METHOD
#   1. Estimate signed curvature along the line.
#   2. Identify corner segments where |R| is between --min-radius and
#      --max-radius, i.e. real corners rather than hairpins or straights.
#   3. Within each segment, apply a lateral offset that is outward at entry and
#      exit and inward at the apex, scaled by --amount and tapered so the line
#      stays continuous.
#   4. Clip the result against the occupancy map, so no waypoint is pushed into
#      a wall, and inset by --margin.
#   5. Recompute curvature and re-derive the velocity profile with the same
#      three limits the original used: lateral grip, forward acceleration and
#      backward deceleration.
#
# USAGE
#   python3 widen_corners.py ~/mapdata/raceline_v7_r.csv \
#       ~/mapdata/bridge_ips_cl.yaml -o ~/mapdata/raceline_v16.csv
#
#   Tune with --amount (metres of widening) and re-run. Start small: 0.12 is
#   about a tenth of the corridor width.
#
# DEPENDENCIES: numpy, Pillow
# =============================================================================

import argparse
import sys

import numpy as np
from PIL import Image


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


def curvature(x, y):
    """Signed curvature of a closed path, positive for a left turn."""
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    denom = (dx * dx + dy * dy) ** 1.5
    denom = np.where(denom < 1e-9, 1e-9, denom)
    return (dx * ddy - dy * ddx) / denom


def smooth_periodic(v, k):
    if k < 2:
        return v
    n = len(v)
    ker = np.ones(k) / k
    return np.convolve(np.r_[v, v, v], ker, 'same')[n:2 * n]


def velocity_profile(x, y, kappa, a_lat, a_acc, a_dec, v_max, v_min):
    n = len(x)
    seg = np.hypot(np.diff(np.append(x, x[0])), np.diff(np.append(y, y[0])))
    with np.errstate(divide='ignore'):
        v = np.sqrt(a_lat / np.maximum(np.abs(kappa), 1e-6))
    v = np.clip(v, v_min, v_max)
    # forward pass, twice because the loop is closed
    for _ in range(2):
        for i in range(n):
            j = (i + 1) % n
            v[j] = min(v[j], np.sqrt(v[i] ** 2 + 2 * a_acc * seg[i]))
    # backward pass
    for _ in range(2):
        for i in range(n - 1, -1, -1):
            j = (i + 1) % n
            v[i] = min(v[i], np.sqrt(v[j] ** 2 + 2 * a_dec * seg[i]))
    return v, seg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('raceline')
    ap.add_argument('map_yaml')
    ap.add_argument('-o', '--out', default='raceline_wide.csv')
    ap.add_argument('--amount', type=float, default=0.12,
                    help='metres of widening at corner entry and exit')
    ap.add_argument('--min-radius', type=float, default=2.5,
                    help='below this, the corner is a hairpin - leave it')
    ap.add_argument('--max-radius', type=float, default=25.0,
                    help='above this it is a straight - leave it')
    ap.add_argument('--margin', type=float, default=0.20,
                    help='metres of clearance kept from any wall')
    ap.add_argument('--skip', type=str, default='240:340',
                    help='waypoint range to leave alone, e.g. pose 1')
    ap.add_argument('--a-lat', type=float, default=4.8)
    ap.add_argument('--a-acc', type=float, default=3.0)
    ap.add_argument('--a-dec', type=float, default=2.2)
    ap.add_argument('--v-max', type=float, default=6.5)
    ap.add_argument('--v-min', type=float, default=1.5)
    args = ap.parse_args()

    rows = [l for l in open(args.raceline) if l.strip()]
    hdr = rows[0]
    P = np.array([[float(v) for v in l.split(',')] for l in rows[1:]])
    x, y = P[:, 0].copy(), P[:, 1].copy()
    n = len(x)

    free, res, org = load_map(args.map_yaml)
    H, W = free.shape

    k0 = smooth_periodic(curvature(x, y), 9)
    R0 = 1.0 / np.maximum(np.abs(k0), 1e-6)
    print(f'  {n} waypoints, radius min {R0.min():.2f} '
          f'median {np.median(R0):.2f} m')

    lo, hi = (int(v) for v in args.skip.split(':'))
    corner = (R0 > args.min_radius) & (R0 < args.max_radius)
    corner[lo:hi] = False
    print(f'  {corner.sum()} waypoints in modifiable corners, '
          f'{hi-lo} skipped at {lo}:{hi}')

    # Offset shape: outward where curvature is rising or falling (entry and
    # exit), inward at the apex where |kappa| peaks. Using the derivative of
    # |kappa| gives exactly that, and it integrates to roughly zero so the
    # line does not drift.
    ak = np.abs(k0)
    dak = smooth_periodic(np.gradient(ak), 15)
    if np.max(np.abs(dak)) > 0:
        dak = dak / np.max(np.abs(dak))
    sign = np.sign(k0)
    sign[sign == 0] = 1.0
    offset = np.where(corner, args.amount * dak * sign, 0.0)
    offset = smooth_periodic(offset, 11)

    dx = np.gradient(x)
    dy = np.gradient(y)
    h = np.hypot(dx, dy)
    nx, ny = -dy / h, dx / h          # left normal

    xn = x + offset * nx
    yn = y + offset * ny

    # clip anything pushed too close to a wall
    clipped = 0
    for i in range(n):
        for t in np.linspace(1.0, 0.0, 11):
            px = x[i] + t * offset[i] * nx[i]
            py = y[i] + t * offset[i] * ny[i]
            cx = int((px - org[0]) / res)
            cy = int((py - org[1]) / res)
            ok = 0 <= cx < W and 0 <= cy < H and free[cy, cx]
            if ok:
                # also require margin clearance either side
                m = int(args.margin / res)
                x0, x1 = max(cx - m, 0), min(cx + m + 1, W)
                y0, y1 = max(cy - m, 0), min(cy + m + 1, H)
                if free[y0:y1, x0:x1].all():
                    xn[i], yn[i] = px, py
                    if t < 1.0:
                        clipped += 1
                    break
        else:
            xn[i], yn[i] = x[i], y[i]
            clipped += 1
    print(f'  {clipped} waypoints clipped for wall clearance')

    k1 = smooth_periodic(curvature(xn, yn), 9)
    R1 = 1.0 / np.maximum(np.abs(k1), 1e-6)
    sel = corner
    if sel.any():
        print(f'  corner radius  before {np.median(R0[sel]):.2f} m'
              f'  after {np.median(R1[sel]):.2f} m')

    v, seg = velocity_profile(xn, yn, k1, args.a_lat, args.a_acc,
                              args.a_dec, args.v_max, args.v_min)
    t_new = float(np.sum(seg / np.maximum(v, 0.1)))
    v_old, seg_old = velocity_profile(x, y, k0, args.a_lat, args.a_acc,
                                      args.a_dec, args.v_max, args.v_min)
    t_old = float(np.sum(seg_old / np.maximum(v_old, 0.1)))
    print(f'  predicted lap  before {t_old:.2f} s  after {t_new:.2f} s'
          f'  ({t_new-t_old:+.2f} s)')
    print(f'  speed min {v.min():.2f}  mean {v.mean():.2f}  max {v.max():.2f}')

    out = np.column_stack([xn, yn, v, k1, np.cumsum(seg) - seg[0]])
    with open(args.out, 'w') as f:
        f.write(hdr)
        for r in out:
            f.write(','.join('%.4f' % q for q in r) + '\n')
    print(f'  wrote {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
