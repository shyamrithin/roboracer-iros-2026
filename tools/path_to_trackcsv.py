#!/usr/bin/env python3
# =============================================================================
# path_to_trackcsv.py
# RoboRacer IROS 2026 / CEM Navigators
# =============================================================================
#
# CODE DESCRIPTION
# -----------------------------------------------------------------------------
# Builds the reference-track CSV the TUM global trajectory optimiser expects
#
#     x_m, y_m, w_tr_right_m, w_tr_left_m
#
# from a DRIVEN LAP rather than from the medial axis of the corridor. Widths
# are measured by ray-casting left and right against the occupancy grid,
# perpendicular to the driven heading at every point.
#
# WHY NOT THE SKELETON
#   map_to_trackcsv.py takes the medial axis of the free space. On this track
#   that fails, and not for a fixable reason: the medial axis of a U-turn cuts
#   to the inside and its radius collapses. Measured at the hairpin near
#   y = -15.4, the skeleton centreline turns 26 degrees per 0.11 m step, which
#   is a radius of 0.24 m. The vehicle's own minimum turning radius is
#   L/tan(delta_max) = 0.324/tan(0.5236) = 0.56 m, so the reference line is
#   not driveable at any speed.
#
#   That is why the TUM optimiser rejected it with "at least two spline
#   normals are crossed": adjacent normals converge 0.26 m out while the track
#   claims 0.85 m of width there, so they cross inside the corridor. Capping
#   the widths does not help, because the defect is in the LINE, not the
#   widths. Seven configurations were tried (s_reg, stepsize_prep/reg, input
#   capping, post-resample capping, crossing horizon, two different maps) and
#   all failed for this reason.
#
#   A driven lap cannot have this defect. The vehicle physically drove it, so
#   its radius is everywhere at least 0.56 m, it is smooth, and it closes.
#
# DATA PROVENANCE
#   Poses come from record_map_data.py, which reads ground-truth IPS. This is
#   an OFFLINE tool and must never enter the submitted container. The
#   organisers confirmed on Slack (2026-09-17) that offline map and raceline
#   preparation from IPS is permitted provided the racing stack subscribes
#   only to /lidar at run time.
#
# METHOD
#   1. Split the pose track into laps by detecting returns to the start point.
#   2. Take one representative lap, resample to uniform arc length, and smooth
#      it with a periodic spline. Driven data has scan-to-scan jitter that a
#      tracker would otherwise chase.
#   3. At each point, cast a ray left and right along the normal until the
#      occupancy grid says occupied, and record the distance. These are the
#      true perpendicular half-widths of the corridor at that point.
#   4. Inset both by --margin so the optimiser keeps clear of the barriers.
#
# USAGE
#   python3 path_to_trackcsv.py ~/mapdata/bridge_scans.npz \
#       ~/mapdata/bridge_ips_cl.yaml -o ~/mapdata/bridge_driven.csv
#
# DEPENDENCIES: numpy, scipy, Pillow
# =============================================================================

import argparse
import sys

import numpy as np
from PIL import Image
from scipy.interpolate import splprep, splev


def load_map(yaml_path):
    cfg = {}
    for line in open(yaml_path):
        line = line.strip()
        if not line or line.startswith('#') or ':' not in line:
            continue
        k, v = line.split(':', 1)
        cfg[k.strip()] = v.strip()
    img_name = cfg['image']
    base = yaml_path.rsplit('/', 1)[0] if '/' in yaml_path else '.'
    grid = np.array(Image.open(f'{base}/{img_name}'))
    res = float(cfg['resolution'])
    org = [float(x) for x in cfg['origin'].strip('[]').split(',')[:2]]
    # map_server stores row 0 at the top; flip so row index increases with y
    grid = np.flipud(grid)
    print(f'  map {grid.shape[1]} x {grid.shape[0]} @ {res} m, '
          f'origin ({org[0]:.3f}, {org[1]:.3f})')
    return grid, res, org


def cast(grid, res, org, x, y, dx, dy, max_m, free_val=254):
    """Distance from (x,y) along (dx,dy) until a non-free cell, in metres."""
    H, W = grid.shape
    steps = int(max_m / (res * 0.5))
    for s in range(1, steps + 1):
        d = s * res * 0.5
        cx = int((x + dx * d - org[0]) / res)
        cy = int((y + dy * d - org[1]) / res)
        if cx < 0 or cx >= W or cy < 0 or cy >= H:
            return d
        if grid[cy, cx] != free_val:
            return d
    return max_m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('npz')
    ap.add_argument('map_yaml')
    ap.add_argument('-o', '--out', default='driven_track.csv')
    ap.add_argument('--n', type=int, default=300, help='output points')
    ap.add_argument('--smooth', type=float, default=0.5,
                    help='periodic spline smoothing factor')
    ap.add_argument('--margin', type=float, default=0.12,
                    help='metres subtracted from each half-width')
    ap.add_argument('--max-width', type=float, default=3.0)
    args = ap.parse_args()

    d = np.load(args.npz)
    P = d['poses'][:, :2]
    print(f'  {len(P)} poses')

    # ---- split into laps ---------------------------------------------------
    start = P[0]
    dist_to_start = np.hypot(P[:, 0] - start[0], P[:, 1] - start[1])
    near = dist_to_start < 0.5
    # rising edges of "near start", ignoring the first 200 samples
    edges = [i for i in range(200, len(P))
             if near[i] and not near[i - 1]]
    print(f'  lap boundaries at {edges}')
    if len(edges) < 2:
        print('  could not find two clean lap boundaries; using whole track')
        lap = P
    else:
        # pick the middle lap: most likely to be steady-state
        k = len(edges) // 2
        lap = P[edges[k - 1]:edges[k]]
    print(f'  using {len(lap)} poses, '
          f'{np.hypot(*np.diff(lap, axis=0).T).sum():.2f} m')

    # ---- resample and smooth ----------------------------------------------
    # drop duplicate consecutive points, which splprep will not accept
    keep = np.hypot(*np.diff(lap, axis=0).T) > 1e-4
    lap = np.vstack((lap[0], lap[1:][keep]))

    tck, _ = splprep([lap[:, 0], lap[:, 1]], s=args.smooth, per=True)
    u = np.linspace(0, 1, args.n, endpoint=False)
    x, y = splev(u, tck)
    dx, dy = splev(u, tck, der=1)
    h = np.hypot(dx, dy)
    tx, ty = dx / h, dy / h
    nx, ny = -ty, tx          # left normal

    seg = np.hypot(np.diff(np.append(x, x[0])), np.diff(np.append(y, y[0])))
    print(f'  smoothed line {seg.sum():.2f} m, {args.n} points, '
          f'spacing {seg.mean()*100:.1f} cm')

    # curvature check - this is the thing the skeleton failed
    hd = np.unwrap(np.arctan2(ty, tx))
    dh = np.abs(np.diff(np.append(hd, hd[0] + 2*np.pi)))
    dh = np.minimum(dh, 2*np.pi - dh)
    with np.errstate(divide='ignore'):
        radius = seg / np.maximum(dh, 1e-9)
    print(f'  min radius {radius.min():.3f} m '
          f'(vehicle limit 0.56 m), median {np.median(radius):.2f} m')

    # ---- measure widths by ray casting ------------------------------------
    grid, res, org = load_map(args.map_yaml)
    wl = np.empty(args.n)
    wr = np.empty(args.n)
    for i in range(args.n):
        wl[i] = cast(grid, res, org, x[i], y[i], nx[i], ny[i], args.max_width)
        wr[i] = cast(grid, res, org, x[i], y[i], -nx[i], -ny[i], args.max_width)

    wl = np.maximum(wl - args.margin, 0.05)
    wr = np.maximum(wr - args.margin, 0.05)
    tot = wl + wr
    print(f'  corridor width  min {tot.min():.2f}  median {np.median(tot):.2f}'
          f'  max {tot.max():.2f} m')
    print(f'  offset from centre: mean {np.mean(wl-wr):+.3f} m '
          f'(0 would mean the driven line is central)')

    out = np.column_stack([x, y, wr, wl])
    np.savetxt(args.out, out, delimiter=',', fmt='%.4f',
               header='x_m,y_m,w_tr_right_m,w_tr_left_m')
    print(f'  wrote {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
