#!/usr/bin/env python3
# =============================================================================
# scans_to_map.py
# RoboRacer IROS 2026 / Team 26 CEM Navigators
# =============================================================================
#
# CODE DESCRIPTION
# -----------------------------------------------------------------------------
# Builds an occupancy grid from LiDAR scans recorded with record_map_data.py,
# by projecting every beam into world coordinates using the ground-truth pose
# captured alongside it. No scan matching, no pose graph, no loop closure -
# each scan lands exactly where it was taken.
#
# OFFLINE TOOL. Uses ground-truth IPS and must never enter the submitted
# container. The organisers confirmed on Slack (2026-09-17) that offline map
# construction from IPS is permitted provided the racing stack subscribes only
# to /lidar at run time.
#
# METHOD
#   1. For each scan, drop beams that are non-finite or at/near range_max
#      (no return), and beams closer than a small minimum (self-hits).
#   2. Rotate the remaining endpoints by the vehicle yaw, translate by the
#      vehicle position, and accumulate hits into a grid.
#   3. Separately accumulate FREE space by marking cells along each beam
#      between the sensor and its endpoint. A cell seen free many times and
#      hit rarely is free; the reverse is occupied. This removes the speckle
#      that a hits-only map suffers from.
#   4. Write map_server pgm + yaml.
#
# WHY BOTH HITS AND RAYS
#   A hits-only grid marks a cell occupied if any beam ever ended there, so a
#   single spurious return leaves a permanent obstacle. Counting how often a
#   cell was passed THROUGH gives the evidence to overrule it.
#
# THE LIDAR IS NOT AT THE VEHICLE ORIGIN
#   AutoDRIVE mounts the LiDAR forward of the rear axle. If the map comes out
#   with walls doubled at a constant offset, set --lidar-x to the mounting
#   offset in metres and rebuild. Start at 0 and inspect.
#
# USAGE
#   python3 scans_to_map.py ~/mapdata/bridge_scans.npz -o ~/mapdata/bridge_ips
#   python3 scans_to_map.py ... --res 0.05 --hit-ratio 0.35
#
# DEPENDENCIES: numpy, Pillow
# =============================================================================

import argparse
import sys

import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('npz')
    ap.add_argument('-o', '--out', default='bridge_ips')
    ap.add_argument('--res', type=float, default=0.05,
                    help='grid cell size in metres')
    ap.add_argument('--lidar-x', type=float, default=0.0,
                    help='LiDAR mounting offset forward of the pose origin')
    ap.add_argument('--hit-ratio', type=float, default=0.35,
                    help='cell is occupied if hits/(hits+passes) exceeds this')
    ap.add_argument('--min-range', type=float, default=0.15)
    ap.add_argument('--stride', type=int, default=1,
                    help='use every Nth scan, to speed up a first look')
    args = ap.parse_args()

    d = np.load(args.npz)
    ranges = d['ranges'][::args.stride]
    poses = d['poses'][::args.stride]
    a0 = float(d['angle_min'])
    da = float(d['angle_increment'])
    rmax = float(d['range_max'])

    n_scans, n_beams = ranges.shape
    print(f'{n_scans} scans x {n_beams} beams')
    print(f'angle_min {a0:.4f}  increment {da:.6f}  range_max {rmax:.2f}')

    beam_ang = a0 + da * np.arange(n_beams)

    # ---- world extent, from the endpoints themselves -----------------------
    xs_all, ys_all = [], []
    for i in range(n_scans):
        r = ranges[i]
        good = np.isfinite(r) & (r > args.min_range) & (r < rmax * 0.99)
        if not good.any():
            continue
        px, py, yaw = poses[i]
        ang = beam_ang[good] + yaw
        xs_all.append(px + args.lidar_x*np.cos(yaw) + r[good]*np.cos(ang))
        ys_all.append(py + args.lidar_x*np.sin(yaw) + r[good]*np.sin(ang))
    xs_all = np.concatenate(xs_all)
    ys_all = np.concatenate(ys_all)

    pad = 0.5
    x0, x1 = xs_all.min() - pad, xs_all.max() + pad
    y0, y1 = ys_all.min() - pad, ys_all.max() + pad
    W = int(np.ceil((x1 - x0) / args.res))
    H = int(np.ceil((y1 - y0) / args.res))
    print(f'world extent {x1-x0:.2f} x {y1-y0:.2f} m  ->  {W} x {H} cells')
    print(f'vehicle path x {poses[:,0].min():.2f}..{poses[:,0].max():.2f}'
          f'  y {poses[:,1].min():.2f}..{poses[:,1].max():.2f}')

    hits = np.zeros((H, W), dtype=np.int32)
    passes = np.zeros((H, W), dtype=np.int32)

    def to_cell(x, y):
        return (((x - x0) / args.res).astype(np.int32),
                ((y - y0) / args.res).astype(np.int32))

    for i in range(n_scans):
        r = ranges[i]
        good = np.isfinite(r) & (r > args.min_range) & (r < rmax * 0.99)
        if not good.any():
            continue
        px, py, yaw = poses[i]
        sx = px + args.lidar_x * np.cos(yaw)
        sy = py + args.lidar_x * np.sin(yaw)
        ang = beam_ang[good] + yaw
        rr = r[good]
        ex, ey = sx + rr*np.cos(ang), sy + rr*np.sin(ang)

        cx, cy = to_cell(ex, ey)
        ok = (cx >= 0) & (cx < W) & (cy >= 0) & (cy < H)
        np.add.at(hits, (cy[ok], cx[ok]), 1)

        # sample along each beam, stopping short of the endpoint
        steps = np.arange(0.0, 1.0, args.res / max(rr.max(), 1e-3))
        if steps.size > 400:
            steps = np.linspace(0.0, 1.0, 400, endpoint=False)
        for t in steps:
            fx = sx + (ex - sx) * t
            fy = sy + (ey - sy) * t
            cxx, cyy = to_cell(fx, fy)
            ok = (cxx >= 0) & (cxx < W) & (cyy >= 0) & (cyy < H)
            np.add.at(passes, (cyy[ok], cxx[ok]), 1)

        if i % 500 == 0:
            print(f'  scan {i}/{n_scans}')

    total = hits + passes
    seen = total > 0
    ratio = np.zeros((H, W))
    ratio[seen] = hits[seen] / total[seen]

    grid = np.full((H, W), 205, dtype=np.uint8)          # unknown
    grid[seen & (ratio <= args.hit_ratio)] = 254         # free
    grid[seen & (ratio > args.hit_ratio)] = 0            # occupied

    print(f'cells: free {(grid==254).sum()}  occupied {(grid==0).sum()}'
          f'  unknown {(grid==205).sum()}')

    # map_server expects row 0 at the TOP, with y increasing upward in the
    # map frame, so flip vertically on write.
    Image.fromarray(np.flipud(grid)).save(args.out + '.pgm')
    with open(args.out + '.yaml', 'w') as f:
        f.write(
            f'image: {args.out.split("/")[-1]}.pgm\n'
            f'resolution: {args.res}\n'
            f'origin: [{x0:.4f}, {y0:.4f}, 0.0]\n'
            f'negate: 0\n'
            f'occupied_thresh: 0.65\n'
            f'free_thresh: 0.25\n')
    print(f'wrote {args.out}.pgm and {args.out}.yaml')
    print(f'origin ({x0:.3f}, {y0:.3f}) - this is a REAL map-frame origin, so '
          f'poses from the recording can be overlaid directly.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
