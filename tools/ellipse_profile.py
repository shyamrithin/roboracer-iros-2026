#!/usr/bin/env python3
# =============================================================================
# ellipse_profile.py
# RoboRacer IROS 2026 / CEM Navigators
# =============================================================================
#
# CODE DESCRIPTION
# -----------------------------------------------------------------------------
# Re-derives the velocity profile of an existing raceline using a friction
# ellipse, so that braking and cornering demand share one budget instead of
# being applied independently.
#
# WHY
#   make_raceline.py applies three limits in sequence: lateral grip
#   v = sqrt(a_lat / |kappa|), then a forward acceleration pass, then a
#   backward deceleration pass. Each is correct on its own, but they are
#   uncoupled - the profile happily plans full braking while also asking for
#   full cornering, which no tyre can deliver.
#
#   That uncoupling is exactly why a_dec 3.0 clipped corner entries while
#   a_dec 2.6 did not. At 3.0 the plan brakes deep into the turn-in, where
#   lateral demand is already high; the vehicle runs out of grip and washes
#   wide. Hand-tuning a_dec to 2.6 works around it by braking gently
#   everywhere, including on the straight where there is no lateral demand at
#   all and full braking would be free.
#
#   The ellipse fixes the shape rather than the magnitude:
#
#       (a_lat / A_LAT)^2 + (a_long / A_LONG)^2  <=  1
#
#   so available longitudinal force is
#
#       a_long = A_LONG * sqrt(1 - (a_lat / A_LAT)^2)
#
#   Braking is hard while the line is straight, blends off as curvature
#   builds, and acceleration comes back progressively on exit. That is the
#   "smooth in, smooth out" behaviour a hand-tuned constant cannot express.
#
# WHAT THIS DOES NOT CHANGE
#   The geometry. This takes an existing line and only rewrites column 3, the
#   target speed. Use it on raceline_v20_r.csv, whose geometry is validated
#   over 137 laps with no collisions.
#
# CALIBRATION
#   A_LAT comes from what the vehicle demonstrably holds: a_lat 4.8 runs
#   clean, 5.5 collides, so 4.8 is the working limit. The slip probe measured
#   a yaw-rate ratio of 1.012 and a median lateral acceleration of 4.4 m/s2
#   with peaks to 9.7, confirming the vehicle is not sliding at that level.
#
#   A_LONG_DEC is the drivetrain drag measured by coastdown at 4.3 m/s2. The
#   old profile used 2.2-2.6 because it had to cover the coupled case; with
#   the ellipse, the full value can be used where the line is straight.
#
#   A_LONG_ACC 3.0 matches what the vehicle achieves from the throttle.
#
# USAGE
#   python3 ellipse_profile.py ~/mapdata/raceline_v20_r.csv \
#       -o ~/mapdata/raceline_ell.csv
#
#   --a-lat and --a-dec scale the ellipse. Start at the defaults, which are
#   the measured values, and only raise a_dec if the corner entries hold.
#
# DEPENDENCIES: numpy
# =============================================================================

import argparse
import sys

import numpy as np


def smooth_periodic(v, k):
    if k < 2:
        return v
    n = len(v)
    return np.convolve(np.r_[v, v, v], np.ones(k) / k, 'same')[n:2 * n]


def curvature(x, y):
    dx, dy = np.gradient(x), np.gradient(y)
    ddx, ddy = np.gradient(dx), np.gradient(dy)
    den = (dx * dx + dy * dy) ** 1.5
    den = np.where(den < 1e-9, 1e-9, den)
    return (dx * ddy - dy * ddx) / den


def ellipse_profile(kappa, seg, a_lat, a_acc, a_dec, v_max, v_min, passes=6):
    """Forward and backward passes with a curvature-dependent longitudinal
    budget. Iterated, because the lateral demand at a point depends on the
    speed there, which the passes are still changing."""
    n = len(kappa)
    ak = np.abs(kappa)
    v = np.clip(np.sqrt(a_lat / np.maximum(ak, 1e-6)), v_min, v_max)

    v_ceiling = v.copy()
    for _ in range(passes):
        # Reset to the grip ceiling each pass. Carrying the reduced v forward
        # ratchets the budget down: a slower v lowers lat_used, but the passes
        # only ever decrease v, so the profile never recovers.
        v = v_ceiling.copy()
        # longitudinal budget left over after cornering, at the ceiling speed
        lat_used = np.clip(ak * v * v / a_lat, 0.0, 1.0)
        avail = np.sqrt(np.maximum(1.0 - lat_used ** 2, 0.02))
        acc = a_acc * avail
        dec = a_dec * avail

        for i in range(n):
            j = (i + 1) % n
            v[j] = min(v[j], np.sqrt(v[i] ** 2 + 2 * acc[i] * seg[i]))
        for i in range(n - 1, -1, -1):
            j = (i + 1) % n
            v[i] = min(v[i], np.sqrt(v[j] ** 2 + 2 * dec[i] * seg[i]))
        v = np.clip(v, v_min, v_max)

    return v


def flat_profile(kappa, seg, a_lat, a_acc, a_dec, v_max, v_min):
    """The old uncoupled profile, for comparison."""
    n = len(kappa)
    v = np.clip(np.sqrt(a_lat / np.maximum(np.abs(kappa), 1e-6)), v_min, v_max)
    for _ in range(3):
        for i in range(n):
            j = (i + 1) % n
            v[j] = min(v[j], np.sqrt(v[i] ** 2 + 2 * a_acc * seg[i]))
        for i in range(n - 1, -1, -1):
            j = (i + 1) % n
            v[i] = min(v[i], np.sqrt(v[j] ** 2 + 2 * a_dec * seg[i]))
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('raceline')
    ap.add_argument('-o', '--out', default='raceline_ell.csv')
    ap.add_argument('--a-lat', type=float, default=4.8)
    ap.add_argument('--a-acc', type=float, default=3.0)
    ap.add_argument('--a-dec', type=float, default=4.3,
                    help='full drivetrain drag; the ellipse scales it down '
                         'wherever the line is curved')
    ap.add_argument('--v-max', type=float, default=6.5)
    ap.add_argument('--v-min', type=float, default=1.5)
    ap.add_argument('--compare-dec', type=float, default=2.6,
                    help='a_dec of the profile being replaced, for the '
                         'lap-time comparison only')
    args = ap.parse_args()

    rows = [l for l in open(args.raceline) if l.strip()]
    hdr = rows[0]
    P = np.array([[float(v) for v in l.split(',')] for l in rows[1:]])
    x, y = P[:, 0], P[:, 1]
    n = len(x)

    seg = np.hypot(np.diff(np.append(x, x[0])), np.diff(np.append(y, y[0])))
    k = smooth_periodic(curvature(x, y), 7)
    R = 1.0 / np.maximum(np.abs(k), 1e-6)
    print(f'  {n} waypoints, {seg.sum():.2f} m, '
          f'radius min {R.min():.2f} median {np.median(R):.2f} m')

    v_old = flat_profile(k, seg, args.a_lat, args.a_acc,
                         args.compare_dec, args.v_max, args.v_min)
    v_new = ellipse_profile(k, seg, args.a_lat, args.a_acc,
                            args.a_dec, args.v_max, args.v_min)

    t_old = float(np.sum(seg / np.maximum(v_old, 0.1)))
    t_new = float(np.sum(seg / np.maximum(v_new, 0.1)))
    print(f'  uncoupled a_dec {args.compare_dec}:  {t_old:.2f} s'
          f'   min {v_old.min():.2f}  mean {v_old.mean():.2f} m/s')
    print(f'  friction ellipse:       {t_new:.2f} s'
          f'   min {v_new.min():.2f}  mean {v_new.mean():.2f} m/s')
    print(f'  difference: {t_new - t_old:+.2f} s')

    # where the difference actually is
    d = v_new - v_old
    up = np.argsort(d)[::-1][:5]
    dn = np.argsort(d)[:5]
    print('  biggest speed gains at waypoints:', 
          ', '.join(f'{i} ({d[i]:+.2f} m/s, R={R[i]:.1f})' for i in up[:3]))
    print('  biggest losses at waypoints:     ',
          ', '.join(f'{i} ({d[i]:+.2f} m/s, R={R[i]:.1f})' for i in dn[:3]))

    out = P.copy()
    out[:, 2] = v_new
    with open(args.out, 'w') as f:
        f.write(hdr)
        for r in out:
            f.write(','.join('%.4f' % q for q in r) + '\n')
    print(f'  wrote {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
