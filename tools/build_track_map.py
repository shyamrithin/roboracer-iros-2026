#!/usr/bin/env python3
# =============================================================================
# build_track_map.py
# RoboRacer IROS 2026 / CEM Navigators
# =============================================================================
#
# CODE DESCRIPTION
# -----------------------------------------------------------------------------
# Builds a ROS 2 occupancy grid (.pgm + .yaml) of the Phase 2 competition track
# from an overhead screenshot of the AutoDRIVE Simulator, scaled using distance
# measured by the vehicle's own wheel encoders.
#
# WHY THIS EXISTS
#   slam_toolbox does not converge on this track. The layout is a bridge
#   silhouette with two near-identical towers and two parallel straights, and
#   the scan matcher repeatedly registers against the wrong one: map->odom
#   swings by metres and tens of degrees per lap and the saved map comes out as
#   several rotated copies of the track. Three configurations were tried,
#   including a crawling run at full 56 Hz scan rate. See session notes.
#
# DATA PROVENANCE - read before using the output
#   Two inputs, both permissible:
#     1. An overhead screenshot of the simulator's own rendered view. This is
#        the published front-end, visible to anyone running the simulator.
#     2. Scale from the vehicle's wheel encoders via srm_racer dead_reckoning.
#   NO simulation ground-truth data is used. /autodrive/roboracer_1/ips, /odom
#   and /tf are NOT read, at build time or at run time. Rules section 2.4
#   prohibits "utilizing simulation ground truth data"; this pipeline avoids
#   it. Simulator asset files are not read either - that would be tapping the
#   back end, which the same section calls out as malpractice.
#
# CALIBRATION
#   Measured 2026-09-16 by teleoperating the vehicle along the deck straight:
#     start   car centroid x = 393.3 px, dead_reckoning dist = 0.000 m
#     end     car centroid x = 912.7 px, dead_reckoning dist = 7.981 m
#     => 519.4 px for 7.981 m  =>  0.015366 m/px
#   Cross-checks: track bounding box 1746 x 795 px -> 26.8 x 12.2 m against a
#   stated track envelope of roughly 30 x 10 m; and a 44.9 m measured lap
#   implies a 2921 px driven path, consistent with a corridor centreline.
#
#   CAVEAT: the simulator view is a perspective, not orthographic, projection.
#   This scale is exact along the deck at y ~ 629 px, where it was measured.
#   Features far from that height - the tower peaks sit ~300 px higher - may
#   carry a small scale error. Quantify before trusting tower geometry to
#   better than a few per cent.
#
# METHOD
#   1. Subtract the lighting gradient with a large Gaussian, so the barrier
#      walls can be thresholded despite the scene being much brighter on the
#      right than the left. A global threshold does not work on this image.
#   2. Binary-close the wall mask to bridge antialiasing gaps.
#   3. Label the free space. The driveable corridor is the large connected
#      component that is not the outside background; the two pillar stubs and
#      the four tower triangle interiors come out as separate sealed regions
#      and are marked occupied, since the vehicle cannot enter them.
#   4. Resample to the target resolution and write map_server pgm + yaml.
#
# OUTPUT CONVENTION
#   pgm: 254 = free, 0 = occupied, 205 = unknown (map_server default)
#   yaml origin is the bottom-left corner of the image in map coordinates,
#   placed so that the track's bounding box starts at (0, 0).
#
# USAGE
#   python3 build_track_map.py <screenshot.png> -o ~/mapdata/bridge_track_geo
#
# DEPENDENCIES: numpy, Pillow, scipy
# =============================================================================

import argparse
import sys

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage


M_PER_PX = 0.015366          # measured, see CALIBRATION above
TARGET_RES = 0.05            # m per cell, matches the Porto maps

CROP = (150, 1080, 78, 1920)  # top, bottom, left, right - strips the UI chrome
BLUR_SIGMA = 51
WALL_THRESH = 20
CLOSE_KERNEL = 7


def extract(path):
    """Return (wall_mask, corridor_mask) at screenshot resolution."""
    grey = np.array(Image.open(path).convert('L')).astype(float)
    t, b, l, r = CROP
    sub = grey[t:b, l:r]

    # The scene is lit from the right, so brightness varies by more across the
    # image than walls differ from floor. Subtract a heavily blurred copy to
    # get a local contrast measure instead of an absolute one.
    bg = np.array(
        Image.fromarray(sub.astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(BLUR_SIGMA))).astype(float)
    wall = (bg - sub) > WALL_THRESH
    wall = ndimage.binary_closing(wall, np.ones((CLOSE_KERNEL, CLOSE_KERNEL)))

    lbl, n = ndimage.label(~wall)
    sizes = ndimage.sum(np.ones_like(lbl), lbl, range(1, n + 1))
    order = np.argsort(sizes)[::-1]

    # Largest free component is the outside background; the corridor is the
    # next largest. Everything else (pillar interiors, tower triangles) is
    # enclosed and unreachable, so it counts as occupied.
    outside = order[0] + 1
    corridor = order[1] + 1
    print(f'  free components: {n}')
    print(f'  outside label {outside}, {int(sizes[outside-1])} px')
    print(f'  corridor label {corridor}, {int(sizes[corridor-1])} px')

    return wall, (lbl == corridor)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('screenshot')
    ap.add_argument('-o', '--out', default='bridge_track_geo')
    ap.add_argument('--res', type=float, default=TARGET_RES)
    ap.add_argument('--m-per-px', type=float, default=M_PER_PX)
    args = ap.parse_args()

    print(f'reading {args.screenshot}')
    wall, corridor = extract(args.screenshot)

    ys, xs = np.nonzero(corridor)
    pad = int(round(0.5 / args.m_per_px))          # 0.5 m margin
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, corridor.shape[1] - 1)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, corridor.shape[0] - 1)
    corr = corridor[y0:y1 + 1, x0:x1 + 1]

    h_px, w_px = corr.shape
    w_m, h_m = w_px * args.m_per_px, h_px * args.m_per_px
    print(f'  corridor bbox {w_px} x {h_px} px  ->  {w_m:.2f} x {h_m:.2f} m')

    # corridor width, from the distance transform of the free space
    dt = ndimage.distance_transform_edt(corr) * args.m_per_px
    inner = dt[corr]
    print(f'  corridor half-width  median {np.median(inner):.3f} m'
          f'  p90 {np.percentile(inner, 90):.3f} m'
          f'  max {inner.max():.3f} m')
    print(f'  => full width        median {2*np.median(inner):.2f} m'
          f'  max {2*inner.max():.2f} m')

    # resample to the target cell size
    scale = args.m_per_px / args.res
    out_w, out_h = int(round(w_px * scale)), int(round(h_px * scale))
    img = Image.fromarray((corr * 255).astype(np.uint8)).resize(
        (out_w, out_h), Image.NEAREST)
    grid = np.array(img) > 127

    pgm = np.where(grid, 254, 0).astype(np.uint8)
    Image.fromarray(pgm).save(args.out + '.pgm')

    with open(args.out + '.yaml', 'w') as fh:
        fh.write(
            f'image: {args.out.split("/")[-1]}.pgm\n'
            f'resolution: {args.res}\n'
            f'origin: [0.0, 0.0, 0.0]\n'
            f'negate: 0\n'
            f'occupied_thresh: 0.65\n'
            f'free_thresh: 0.25\n')

    print(f'  wrote {args.out}.pgm  {out_w} x {out_h} cells @ {args.res} m')
    print(f'  wrote {args.out}.yaml')
    return 0


if __name__ == '__main__':
    sys.exit(main())
