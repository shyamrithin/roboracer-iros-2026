#!/usr/bin/env python3
# =============================================================================
# map_to_trackcsv.py
# -----------------------------------------------------------------------------
# Project : RoboRacer Sim Racing League @ IROS 2026 - Team CEM Navigators (26)
# Author  : Shyam Rithin
# Location: ~/roboracer/tools/map_to_trackcsv.py
# Usage   : python3 map_to_trackcsv.py ~/mapdata/porto.yaml -o porto_track.csv
# =============================================================================
#
# DESCRIPTION
# -----------
# Turns a slam_toolbox occupancy grid into the reference-track CSV that the TUM
# global trajectory optimiser expects:
#
#     x_m, y_m, w_tr_right_m, w_tr_left_m
#
# a centreline with the distance to the wall on each side at every point.
#
# WHY THIS EXISTS
# ---------------
# The raceline used so far was a driven lap, smoothed. A driven lap is not an
# optimal line: the disparity extender aims at the deepest gap, which in a
# corner is the inside, so the recorded line hugs the inner wall and its
# minimum radius stays near the vehicle's limit. That is what caps the
# tracker's speed - at scale 0.7 the two tightest sections are where tracking
# error quadruples and the collisions happen.
#
# A minimum-curvature optimiser uses the whole corridor: wide entry, late
# apex, wide exit. That raises the minimum radius, which makes the line both
# faster and easier to track. Those usually trade against each other; here
# they do not.
#
# The optimiser needs track boundaries, not a driven path. This script
# produces them.
#
# METHOD
# ------
#   1. Free space is thresholded out of the grid using the occupancy
#      thresholds in the map yaml.
#   2. A Euclidean distance transform gives, for every free pixel, the
#      distance to the nearest wall. On the centreline that distance IS the
#      track half-width, which is why the two are computed together.
#   3. The free space is skeletonised. The skeleton of a closed corridor is
#      its centreline.
#   4. Spurs are pruned. Skeletonisation of a noisy grid throws off short
#      branches at wall irregularities; repeatedly deleting degree-1 pixels
#      removes them and leaves the cycle.
#   5. The cycle is walked in order and resampled to uniform spacing.
#
# The half-width is reported symmetrically. On a true centreline the left and
# right distances are equal by construction, and the optimiser only needs the
# corridor, not which side is which. The sample tracks shipped with the
# optimiser do the same.
#
# OUTPUT SANITY
# -------------
# The printed half-width range is the check that matters. Porto's corridor is
# about 2.6 m, so half-widths should land near 1.3 m. If they come out at tens
# of metres the skeleton has escaped into unmapped space and the map needs
# cropping first - one of the optimiser's own sample tracks has 25 m
# half-widths from exactly that failure.
# =============================================================================

import argparse
import math
import os
import sys

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize


def read_yaml(path):
    """Minimal reader for the map yaml. Avoids a pyyaml dependency."""
    cfg = {}
    with open(path) as fh:
        for line in fh:
            line = line.split("#")[0].strip()
            if not line or ":" not in line:
                continue
            k, v = line.split(":", 1)
            k, v = k.strip(), v.strip()
            if v.startswith("["):
                cfg[k] = [float(x) for x in v.strip("[]").split(",")]
            else:
                try:
                    cfg[k] = float(v)
                except ValueError:
                    cfg[k] = v
    return cfg


def read_pgm(path):
    """Read a binary (P5) or plain (P2) PGM into a numpy array."""
    with open(path, "rb") as fh:
        data = fh.read()

    # Header tokens, skipping comments.
    tokens, i = [], 0
    while len(tokens) < 4:
        while i < len(data) and data[i : i + 1].isspace():
            i += 1
        if data[i : i + 1] == b"#":
            while i < len(data) and data[i] != 0x0A:
                i += 1
            continue
        j = i
        while j < len(data) and not data[j : j + 1].isspace():
            j += 1
        tokens.append(data[i:j])
        i = j
    i += 1  # single whitespace after maxval

    magic = tokens[0]
    w, h, maxval = int(tokens[1]), int(tokens[2]), int(tokens[3])

    if magic == b"P5":
        dtype = np.uint8 if maxval < 256 else ">u2"
        img = np.frombuffer(data, dtype=dtype, count=w * h, offset=i)
        return img.reshape(h, w).astype(np.float64)
    if magic == b"P2":
        vals = np.array(data[i:].split(), dtype=np.float64)
        return vals[: w * h].reshape(h, w)
    raise ValueError(f"unsupported PGM magic {magic!r}")


def neighbours(r, c):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr or dc:
                yield r + dr, c + dc


def prune_spurs(skel, max_passes=200):
    """Delete degree-1 pixels repeatedly until only the cycle remains."""
    sk = skel.copy()
    for _ in range(max_passes):
        pts = set(zip(*np.nonzero(sk)))
        ends = [p for p in pts
                if sum((n in pts) for n in neighbours(*p)) <= 1]
        if not ends:
            break
        if len(ends) == len(pts):
            break                      # degenerate: nothing is a loop
        for r, c in ends:
            sk[r, c] = False
    return sk


def order_cycle(sk):
    """Walk the skeleton cycle, returning pixels in traversal order."""
    pts = set(zip(*np.nonzero(sk)))
    if not pts:
        return []
    start = min(pts)
    order = [start]
    used = {start}
    cur = start
    while True:
        nxt = None
        # Prefer 4-connected steps so the walk does not cut corners.
        cands = [n for n in neighbours(*cur) if n in pts and n not in used]
        if cands:
            cands.sort(key=lambda p: abs(p[0] - cur[0]) + abs(p[1] - cur[1]))
            nxt = cands[0]
        if nxt is None:
            break
        order.append(nxt)
        used.add(nxt)
        cur = nxt
    return order


def resample_closed(x, y, w, n):
    """Resample a closed polyline to n points of uniform arc length."""
    xc = np.append(x, x[0])
    yc = np.append(y, y[0])
    wc = np.append(w, w[0])
    seg = np.hypot(np.diff(xc), np.diff(yc))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    su = np.linspace(0.0, total, n, endpoint=False)
    return (np.interp(su, s, xc), np.interp(su, s, yc),
            np.interp(su, s, wc), total)


def smooth_closed(a, window):
    if window < 3:
        return a.copy()
    if window % 2 == 0:
        window += 1
    k = np.ones(window) / window
    p = np.concatenate([a[-window:], a, a[:window]])
    return np.convolve(p, k, mode="same")[window:-window]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("yaml", help="map yaml, e.g. ~/mapdata/porto.yaml")
    ap.add_argument("-o", "--out", default="track.csv")
    ap.add_argument("--n", type=int, default=400, help="output waypoints")
    ap.add_argument("--smooth", type=int, default=9,
                    help="centreline smoothing window")
    ap.add_argument("--unknown-value", type=float, default=205.0,
                    help="pixel value meaning unknown; excluded from free")
    ap.add_argument("--unknown-tol", type=float, default=12.0)
    ap.add_argument("--margin", type=float, default=0.0,
                    help="metres to subtract from each half-width, as a "
                         "safety inset")
    args = ap.parse_args()

    ypath = os.path.expanduser(args.yaml)
    if not os.path.isfile(ypath):
        print(f"error: no such file: {ypath}")
        sys.exit(1)

    cfg = read_yaml(ypath)
    res = cfg["resolution"]
    ox, oy = cfg["origin"][0], cfg["origin"][1]
    occ_th = cfg.get("occupied_thresh", 0.65)
    free_th = cfg.get("free_thresh", 0.25)
    negate = int(cfg.get("negate", 0))

    img_name = cfg["image"]
    ipath = img_name if os.path.isabs(img_name) else os.path.join(
        os.path.dirname(ypath), img_name)

    print(f"\nmap_to_trackcsv.py  --  {ypath}")
    img = read_pgm(ipath)
    h, w = img.shape
    print(f"  image {ipath}")
    print(f"  {w} x {h} cells at {res} m  ->  {w*res:.2f} x {h*res:.2f} m")
    print(f"  origin ({ox}, {oy})")

    # Occupancy per the map_server convention.
    p = img / 255.0
    occ = p if negate else (1.0 - p)
    # Unknown cells must NOT count as free. slam_toolbox writes unknown as
    # 205, which is occupancy 0.196 - below this map's free_thresh of 0.25,
    # so a naive threshold swallows the infield and everything outside the
    # circuit, and the skeleton is then the medial axis of the whole image
    # rather than of the corridor.
    unknown = np.abs(img - args.unknown_value) <= args.unknown_tol
    free = (occ < free_th) & ~unknown
    wall = (occ > occ_th) | unknown
    vals, cnts = np.unique(img.astype(np.int64), return_counts=True)
    top = sorted(zip(cnts, vals), reverse=True)[:4]
    print("  pixel histogram (count, value): " +
          ", ".join(f"({c}, {v})" for c, v in top))
    print(f"  free {free.sum()} cells, occupied {wall.sum()}, "
          f"unknown {img.size - free.sum() - wall.sum()}")

    # Distance to the nearest non-free cell. Unknown counts as non-free, which
    # keeps the skeleton inside the mapped corridor.
    dist_px = ndimage.distance_transform_edt(free)
    dist_m = dist_px * res

    skel = skeletonize(free)
    print(f"  skeleton {int(skel.sum())} cells before pruning")
    skel = prune_spurs(skel)
    print(f"  skeleton {int(skel.sum())} cells after pruning")
    if skel.sum() < 20:
        print("\nerror: the skeleton collapsed. The free space is probably "
              "not a closed\nloop - crop the map to the circuit and retry.")
        sys.exit(1)

    order = order_cycle(skel)
    print(f"  walked {len(order)} of {int(skel.sum())} skeleton cells")
    if len(order) < 0.8 * skel.sum():
        print("  warning: the walk missed a fifth of the skeleton, so the "
              "centreline\n  may be broken. Inspect the output before "
              "optimising.")

    rows = np.array([p[0] for p in order])
    cols = np.array([p[1] for p in order])

    # Grid to world. Row 0 is the TOP of the image, so y is flipped.
    X = ox + (cols + 0.5) * res
    Y = oy + (h - 1 - rows + 0.5) * res
    W = dist_m[rows, cols]

    X = smooth_closed(X, args.smooth)
    Y = smooth_closed(Y, args.smooth)
    W = smooth_closed(W, args.smooth)

    X, Y, W, total = resample_closed(X, Y, W, args.n)
    W = np.maximum(W - args.margin, 0.05)

    print(f"\n  centreline {args.n} points, {total:.2f} m, "
          f"spacing {total/args.n*100:.1f} cm")
    print(f"  extent x [{X.min():.2f},{X.max():.2f}]  "
          f"y [{Y.min():.2f},{Y.max():.2f}]")
    print(f"  half-width min {W.min():.2f}  mean {W.mean():.2f}  "
          f"max {W.max():.2f} m")
    print(f"  -> corridor {2*W.mean():.2f} m wide on average")

    if W.max() > 5.0:
        print("\n  WARNING: half-widths above 5 m. The skeleton has almost "
              "certainly\n  escaped into open or unmapped space. Crop the map "
              "before optimising.")

    out = os.path.expanduser(args.out)
    with open(out, "w") as fh:
        fh.write("# x_m,y_m,w_tr_right_m,w_tr_left_m\n")
        for i in range(args.n):
            fh.write(f"{X[i]:.4f},{Y[i]:.4f},{W[i]:.4f},{W[i]:.4f}\n")
    print(f"\nwrote {out}")
    print("\nOverlay this on the map before optimising. The centreline should "
          "sit\nmidway between the walls the whole way round.")


if __name__ == "__main__":
    main()
