#!/usr/bin/env python3
# =============================================================================
# patch_depth_v2.py
# -----------------------------------------------------------------------------
# Project : RoboRacer Sim Racing League @ IROS 2026 - Team CEM Navigators (26)
# Author  : Shyam Rithin
# Usage   : python3 patch_depth_v2.py ~/roboracer/devkit_src/srm_racer/srm_racer/gap_follower.py
# =============================================================================
#
# DESCRIPTION
# -----------
# Adds a corrected path-depth measure (_path_depth_v2) alongside the existing
# one, and logs both. The CONTROL PATH IS UNCHANGED: throttle still uses the
# original measure, and use_depth_throttle stays false. This is the
# diagnostic-before-wiring step that the three earlier depth attempts skipped.
#
# WHAT THE FULL-RATE LOG SHOWED ABOUT THE ORIGINAL
#   median 8.00 m whenever |steer| > 0.40, with 52.5 per cent of those samples
#   pinned at arc_max; 14.7 per cent of straight-line samples below 0.30 m;
#   correlation with |steer| of +0.226 where a working measure must be
#   strongly negative. The measure is inverted: it reports maximum clearance
#   exactly when the vehicle is turning hardest.
#
# THREE DEFECTS, AND WHAT v2 DOES ABOUT EACH
#
# 1. COORDINATE TRANSFORM (the cause of the near-zero readings).
#    _prepare subtracts the 0.141 m LiDAR-to-bumper offset from each RANGE,
#    and _path_depth then treats the shrunk range as a polar coordinate:
#        x = (r_raw - 0.141) cos(theta),  y = (r_raw - 0.141) sin(theta)
#    But the bumper offset is a TRANSLATION ALONG X, not a radial shrink. The
#    two agree only at theta = 0 and diverge as bearing grows. A return at
#    0.18 m and 50 deg is genuinely 2.5 cm BEHIND the bumper plane; the old
#    maths places it 2.5 cm AHEAD, so it is counted as an obstacle directly in
#    the path and drags the minimum to nothing.
#
#    v2 reconstructs the true position relative to the bumper:
#        x = r_raw cos(theta) - 0.141,  y = r_raw sin(theta)
#    and discards everything with x <= 0, which cannot be ahead of the car.
#
# 2. INSTANTANEOUS CURVATURE USED AS AN 8 M PREVIEW.
#    R = L / tan(delta) gives 1.16 m at the 52 per cent lock this vehicle
#    actually reaches. That circle is 2.33 m across and the corridor measures
#    2.61 m mean, so the circle fits inside without touching either wall. The
#    old code then correctly reports no constraint. The geometry is right; the
#    assumption is wrong, because the vehicle will not drive that circle - the
#    steering changes on the next scan.
#
#    v2 floors the projection radius at min_preview_radius_m. A gentler arc is
#    a better model of the average curvature over a multi-metre preview than
#    the instantaneous value, and it guarantees the arc leaves the corridor so
#    a wall is always found.
#
# 3. NO GUARD ON THE VEHICLE'S OWN FOOTPRINT.
#    Even with the transform fixed, returns from the front wheels or very
#    close bodywork could constrain the path. v2 ignores anything closer than
#    min_arc_m along the arc.
#
# NEW PARAMETERS (both live-tunable)
#   min_preview_radius_m   default 3.0   floor on the projection radius
#   min_arc_m              default 0.25  ignore constraints closer than this
#
# HOW TO READ THE RESULTING LOG
#   A working measure must show depth FALLING as |steer| rises, i.e. a
#   strongly NEGATIVE correlation, no pile-up at arc_max during hard steering,
#   and no near-zero values while running straight. If d2 does not show that,
#   it is no better than d1 and must not be wired to throttle.
# =============================================================================

import sys
import os

NEW_METHOD = '''
    def _path_depth_v2(self, ranges, angles, steer_norm):
        """
        Corrected free distance along the arc the vehicle is following.

        Differs from _path_depth in three ways; see the revision notes for the
        evidence behind each.

        First, the bumper offset is treated as a translation rather than a
        radial shrink. _prepare returns bumper-referenced ranges, so the raw
        range is recovered before converting to Cartesian and the offset is
        then applied along x only. Points with x <= 0 lie behind the bumper
        plane and are discarded rather than counted as obstacles ahead.

        Second, the projection radius is floored. The instantaneous radius
        R = L / tan(delta) reaches 1.16 m at the lock this vehicle actually
        uses, and a circle that tight fits inside the corridor without
        touching it, so no constraint is found and the measure saturates. Over
        a preview of several metres the steering will change substantially, so
        a floored radius is the better model of average curvature.

        Third, constraints closer than min_arc_m are ignored, so the vehicle's
        own wheels and bodywork cannot pin the measure at zero.
        """
        # --- true Cartesian relative to the front bumper --------------------
        r_raw = ranges + LIDAR_TO_BUMPER_M
        xs = r_raw * np.cos(angles) - LIDAR_TO_BUMPER_M
        ys = r_raw * np.sin(angles)

        ahead = xs > 0.0
        if not ahead.any():
            return self.arc_max_m

        delta = steer_norm * MAX_STEER_RAD
        tan_delta = math.tan(delta)

        if abs(tan_delta) < 1e-3:
            on_path = ahead & (np.abs(ys) <= self.path_half_width_m)
            if not on_path.any():
                return self.arc_max_m
            d = xs[on_path]
            d = d[d >= self.min_arc_m]
            if d.size == 0:
                return self.arc_max_m
            return float(np.clip(np.min(d), 0.0, self.arc_max_m))

        radius = WHEELBASE_M / tan_delta          # signed; left turn positive
        if abs(radius) < self.min_preview_radius_m:
            radius = math.copysign(self.min_preview_radius_m, radius)
        centre_y = radius

        offset = np.abs(np.hypot(xs, ys - centre_y) - abs(radius))
        on_path = ahead & (offset <= self.path_half_width_m)

        phi = np.arctan2(xs, np.sign(radius) * (centre_y - ys))
        valid = on_path & (phi > 0.0)
        if not valid.any():
            return self.arc_max_m

        arc_len = abs(radius) * phi[valid]
        arc_len = arc_len[arc_len >= self.min_arc_m]
        if arc_len.size == 0:
            return self.arc_max_m
        return float(np.clip(np.min(arc_len), 0.0, self.arc_max_m))

    def _aim_from_steer(self, steer_norm):
        """
        Recover the aim bearing from the steering command.

        Pure pursuit gives delta = atan(2 L sin(alpha) / Ld), so
        sin(alpha) = tan(delta) Ld / (2 L). Inverting here avoids changing
        the scan_callback signature. Verified against logged telemetry: a
        command of 0.519 recovers 1.036 rad against a logged aim of 1.034.
        """
        if not self.use_pure_pursuit:
            return steer_norm * MAX_STEER_RAD / max(self.steering_gain, 1e-6)
        delta = steer_norm * MAX_STEER_RAD
        s = math.tan(delta) * self.lookahead_m / (2.0 * WHEELBASE_M)
        return float(math.asin(float(np.clip(s, -1.0, 1.0))))

    def _depth_cone(self, ranges, angles, centre_rad, half_width_rad):
        """
        Minimum bumper-referenced range inside a wedge about a bearing.

        Used for two of the candidate measures: a narrow wedge straight ahead,
        which is the quantity the simulator HUD reports as a single LiDAR
        measurement, and a wedge about the aim bearing.
        """
        mask = np.abs(angles - centre_rad) <= half_width_rad
        if not mask.any():
            return self.arc_max_m
        return float(np.clip(np.min(ranges[mask]), 0.0, self.arc_max_m))

    def _path_depth(self, ranges, angles, steer_norm):
        """
        Comparison wrapper. Computes four candidate depth measures, stores
        three of them for logging, and returns the ORIGINAL so the control
        path is unchanged while all four are compared on the same runs.

          d1  original arc projection
          d2  corrected arc projection
          dc  narrow wedge straight ahead
          da  wedge about the aim bearing

        A usable measure must fall as the corridor closes AND lead the
        steering, so that throttle is already dropping while the corner is
        still several metres away. Deceleration is by idle torque alone.
        """
        self.depth_v2 = self._path_depth_v2(ranges, angles, steer_norm)
        self.depth_cone = self._depth_cone(
            ranges, angles, 0.0, self.centre_cone_rad)
        self.depth_aim = self._depth_cone(
            ranges, angles, self._aim_from_steer(steer_norm),
            self.aim_cone_rad)
        return self._path_depth_v1(ranges, angles, steer_norm)
'''


def main():
    if len(sys.argv) < 2:
        print("usage: python3 patch_depth_v2.py <path to gap_follower.py>")
        sys.exit(1)
    path = os.path.expanduser(sys.argv[1])
    if not os.path.isfile(path):
        print(f"error: no such file: {path}")
        sys.exit(1)

    s = open(path).read()
    orig = s

    # ---- 1. new parameters ------------------------------------------------
    anchor = "        self.declare_parameter('arc_max_m', 8.0)"
    if anchor not in s:
        print("error: could not find the arc_max_m declaration")
        sys.exit(1)
    if "min_preview_radius_m" in s:
        print("error: already patched")
        sys.exit(1)
    s = s.replace(anchor, anchor
                  + "\n        self.declare_parameter('min_preview_radius_m', 3.0)"
                  + "\n        self.declare_parameter('min_arc_m', 0.25)"
                  + "\n        self.declare_parameter('centre_cone_deg', 4.0)"
                  + "\n        self.declare_parameter('aim_cone_deg', 6.0)")

    anchor = "        self.arc_max_m = g('arc_max_m').value"
    s = s.replace(anchor, anchor
                  + "\n        self.min_preview_radius_m = g('min_preview_radius_m').value"
                  + "\n        self.min_arc_m = g('min_arc_m').value"
                  + "\n        self.centre_cone_rad = math.radians(g('centre_cone_deg').value)"
                  + "\n        self.aim_cone_rad = math.radians(g('aim_cone_deg').value)")

    # ---- 2. rename the original, insert v2 and the wrapper ----------------
    anchor = "    def _path_depth(self, ranges, angles, steer_norm):"
    if anchor not in s:
        print("error: could not find _path_depth")
        sys.exit(1)
    s = s.replace(anchor,
                  "    def _path_depth_v1(self, ranges, angles, steer_norm):", 1)

    anchor = "    def _throttle_for(self, depth_m, steer_norm):"
    if anchor not in s:
        print("error: could not find _throttle_for")
        sys.exit(1)
    s = s.replace(anchor, NEW_METHOD.strip("\n") + "\n\n" + anchor, 1)

    # ---- 3. initialise the attribute --------------------------------------
    anchor = "        self.last_log_s = 0.0"
    if anchor in s:
        s = s.replace(anchor, anchor
                      + "\n        self.depth_v2 = 0.0"
                      + "\n        self.depth_cone = 0.0"
                      + "\n        self.depth_aim = 0.0", 1)

    # ---- 4. log both measures ---------------------------------------------
    old_fmt = ("            'aim={:+.3f}  steer={:+.3f}  thr={:.3f}  depth={:.2f}  '\n"
               "            'L={:.2f} R={:.2f} bias={:+.3f}'.format(\n"
               "                target_rad, steer_norm, throttle, depth_m,\n"
               "                free_l, free_r, bias_rad))")
    new_fmt = ("            'aim={:+.3f}  steer={:+.3f}  thr={:.3f}  '\n"
               "            'd1={:.2f} d2={:.2f} dc={:.2f} da={:.2f}  '\n"
               "            'L={:.2f} R={:.2f} bias={:+.3f}'.format(\n"
               "                target_rad, steer_norm, throttle,\n"
               "                depth_m, self.depth_v2, self.depth_cone,\n"
               "                self.depth_aim,\n"
               "                free_l, free_r, bias_rad))")
    if old_fmt in s:
        s = s.replace(old_fmt, new_fmt, 1)
    else:
        print("WARNING: log format string not matched; patching parameters and")
        print("         methods only. Add self.depth_v2 to the log by hand.")

    if s == orig:
        print("error: nothing changed")
        sys.exit(1)

    open(path, "w").write(s)
    print(f"patched {path}")
    print("  + min_preview_radius_m (3.0), min_arc_m (0.25),")
    print("    centre_cone_deg (4.0), aim_cone_deg (6.0)")
    print("  + _path_depth_v2, original renamed to _path_depth_v1")
    print("  + log now prints d1 d2 dc da")
    print("  control path UNCHANGED: throttle still uses d1, "
          "use_depth_throttle still false")


if __name__ == "__main__":
    main()
