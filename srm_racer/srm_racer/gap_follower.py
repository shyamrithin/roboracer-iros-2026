#!/usr/bin/env python3
# =============================================================================
# File        : gap_follower.py
# Package     : srm_racer
# Project     : RoboRacer Sim Racing League @ IROS 2026 (AutoDRIVE Ecosystem)
# Target      : ROS 2 Humble / Python 3.10
# Created     : 2026-08-31
# Revised     : 2026-09-04  (v7: clean canonical disparity extender)
# Revised     : 2026-09-04  (v8: pure pursuit steering law, central tie-break
#                            among near-maximal headings)
# =============================================================================
# CODE DESCRIPTION
# -----------------------------------------------------------------------------
# Canonical disparity extender for the RoboRacer digital twin in the AutoDRIVE
# Simulator. Consumes only the 2D LiDAR stream, which is a permissible input at
# run time, and emits normalised steering and throttle commands. No map, no
# stored knowledge of the circuit, and no ground-truth pose, so the behaviour
# transfers unchanged to an unseen racetrack. The core method is due to
# Otterness (UNC-Chapel Hill, 2019).
#
# REVISION NOTES (v8)
# -----------------------------------------------------------------------------
# Two defects in v7 are addressed, and each is separately switchable so their
# effects can be measured independently.
#
# 1. LINEAR BEARING TO STEERING MAPPING.
#    v7 set the steering command to bearing / max_steer. A bearing of 50 deg
#    then demands 167 per cent of the mechanical limit and clips to full lock.
#    Logged output showed the command saturated at -1.000 through every corner.
#    Scaling the whole response down by a constant (steering_gain) suppressed
#    the saturation but also removed authority where a genuinely sharp
#    correction was needed, so the vehicle failed to recover after deep turns.
#
#    v8 restores the pure pursuit law
#        delta = atan(2 * L * sin(alpha) / Ld)
#    which compresses wide bearings without weakening the response to small
#    ones. It is the principled version of what a constant gain approximates.
#    Setting `use_pure_pursuit` false reverts to the v7 linear mapping.
#
# 2. ARGMAX PINNING AT THE FIELD OF VIEW BOUNDARY.
#    Logged aim bearings sat at exactly -0.873 rad with a 50 deg half-window,
#    and at exactly -1.571 rad with a 90 deg one, repeatedly. Those are the
#    window edges to the sample. Two causes combine: the deepest return
#    genuinely lies off to the side when the corridor turns, and np.diff cannot
#    detect a disparity at the first or last sample because that sample has no
#    neighbour inside the window, so a boundary reading is never extended.
#
#    v8 breaks ties centrally. Rather than taking the single deepest sample, it
#    collects every sample within `tie_tolerance_m` of the maximum and selects
#    the one nearest straight ahead. When several headings are effectively
#    equally open, the most forward of them is preferred, which stops the aim
#    drifting to a boundary that merely happens to read marginally longer.
#    Setting `tie_tolerance_m` to zero restores exact v7 argmax behaviour.
#
# ALGORITHM, once per incoming laser scan:
#   1. Restrict the scan to a forward field of view and clip ranges to a
#      horizon; treat non-returns as the horizon, since they denote open space.
#   2. Find disparities, meaning adjacent samples differing by more than a
#      threshold. Each marks an obstacle edge.
#   3. At each disparity, overwrite samples on the far side with the nearer
#      reading, across the angular span half a car width subtends at that
#      range. The LiDAR reasons as a single point but sits on a vehicle 0.27 m
#      wide, so a genuinely visible distant point can still put a wheel or
#      flank into the obstacle edge preceding it. Extending the disparity
#      removes every heading the vehicle cannot physically take.
#   4. Select the deepest remaining direction, preferring the most forward
#      among near-equal candidates.
#   5. Convert that bearing to a steering angle by pure pursuit.
#   6. Drive at a constant throttle.
#
# SPEED REMAINS CONSTANT IN THIS REVISION, deliberately. The purpose is still
# to establish that the steering geometry alone completes clean laps before any
# speed dynamics are reintroduced.
#
# DESIGN CONSTRAINTS
#   * Negative throttle engages REVERSE on this vehicle rather than braking, so
#     throttle is clamped non-negative at all times.
#   * Restricted topics (ips, odom, tf, lap and collision telemetry, and
#     reset_command) are never subscribed to, per section 2.4 of the rule book.
#   * The bridge tick observed on the development laptop is ~18.5 Hz against a
#     documented 40 Hz sensor rate, and the evaluation workstation will differ.
#     This revision holds no rate-dependent state, so it is unaffected.
#
# GEOMETRY (2026 Technical Guide)
#   Car width 0.270 m, wheelbase 0.324 m, front overhang 0.090 m.
#   LiDAR frame at x = 0.2733 m from the rear axle; front bumper at 0.414 m,
#   hence obstacles are 0.141 m closer to the bumper than the LiDAR reports.
#   Steering limits +/- 0.5236 rad. Scan is 1080 points over 270 degrees.
# =============================================================================

import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32

# --- Fixed vehicle geometry, do not tune -------------------------------------
LIDAR_TO_BUMPER_M = 0.141      # 0.414 m bumper - 0.2733 m LiDAR mount
CAR_HALF_WIDTH_M = 0.135       # 0.270 m overall width
WHEELBASE_M = 0.324
MAX_STEER_RAD = 0.5236         # +/- 30 deg mechanical limit


class GapFollower(Node):
    """Canonical disparity extender with pure pursuit steering."""

    def __init__(self):
        super().__init__('gap_follower')

        self.declare_parameter('fov_deg', 60.0)
        self.declare_parameter('horizon_m', 8.0)
        self.declare_parameter('disparity_threshold_m', 0.20)
        self.declare_parameter('extend_margin_m', 0.16)
        self.declare_parameter('tie_tolerance_m', 0.50)
        self.declare_parameter('use_pure_pursuit', True)
        self.declare_parameter('lookahead_m', 2.00)
        self.declare_parameter('steering_gain', 0.45)
        self.declare_parameter('throttle', 0.22)
        self.declare_parameter('use_depth_throttle', False)
        self.declare_parameter('throttle_min', 0.09)
        self.declare_parameter('throttle_max', 0.26)
        self.declare_parameter('depth_min_m', 2.0)
        self.declare_parameter('depth_max_m', 6.0)
        self.declare_parameter('front_cone_deg', 15.0)
        self.declare_parameter('steer_derate', 1.0)
        self.declare_parameter('derate_exponent', 1.0)
        self.declare_parameter('path_half_width_m', 0.32)
        self.declare_parameter('arc_max_m', 8.0)
        self.declare_parameter('min_preview_radius_m', 3.0)
        self.declare_parameter('min_arc_m', 0.25)
        self.declare_parameter('centre_cone_deg', 4.0)
        self.declare_parameter('aim_cone_deg', 6.0)
        self.declare_parameter('log_period_s', 1.0)
        self.declare_parameter('bias_side_deg', 55.0)
        self.declare_parameter('bias_window_deg', 25.0)
        self.declare_parameter('bias_gain', 0.0)  # disabled: see _corridor_bias

        self._reload_parameters()
        self.last_log_s = 0.0
        self.depth_v2 = 0.0
        self.depth_cone = 0.0
        self.depth_aim = 0.0

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.steer_pub = self.create_publisher(
            Float32, '/autodrive/roboracer_1/steering_command', 10)
        self.throttle_pub = self.create_publisher(
            Float32, '/autodrive/roboracer_1/throttle_command', 10)
        self.scan_sub = self.create_subscription(
            LaserScan, '/autodrive/roboracer_1/lidar', self.scan_callback, qos)

        self.get_logger().info('gap_follower v8 ready, waiting for laser scans')

    def _reload_parameters(self):
        """Pull current parameter values into plain attributes each cycle."""
        g = self.get_parameter
        self.fov_rad = math.radians(g('fov_deg').value)
        self.horizon_m = g('horizon_m').value
        self.disparity_threshold_m = g('disparity_threshold_m').value
        self.extend_margin_m = g('extend_margin_m').value
        self.tie_tolerance_m = g('tie_tolerance_m').value
        self.use_pure_pursuit = g('use_pure_pursuit').value
        self.lookahead_m = g('lookahead_m').value
        self.steering_gain = g('steering_gain').value
        self.throttle = g('throttle').value
        self.use_depth_throttle = g('use_depth_throttle').value
        self.throttle_min = g('throttle_min').value
        self.throttle_max = g('throttle_max').value
        self.depth_min_m = g('depth_min_m').value
        self.depth_max_m = g('depth_max_m').value
        self.front_cone_rad = math.radians(g('front_cone_deg').value)
        self.steer_derate = g('steer_derate').value
        self.derate_exponent = g('derate_exponent').value
        self.path_half_width_m = g('path_half_width_m').value
        self.arc_max_m = g('arc_max_m').value
        self.min_preview_radius_m = g('min_preview_radius_m').value
        self.min_arc_m = g('min_arc_m').value
        self.centre_cone_rad = math.radians(g('centre_cone_deg').value)
        self.aim_cone_rad = math.radians(g('aim_cone_deg').value)
        self.log_period_s = g('log_period_s').value
        self.bias_side_rad = math.radians(g('bias_side_deg').value)
        self.bias_window_rad = math.radians(g('bias_window_deg').value)
        self.bias_gain = g('bias_gain').value

    def scan_callback(self, msg: LaserScan):
        """Main control loop, executed once per laser scan."""
        self._reload_parameters()

        ranges, angles = self._prepare(msg)
        if ranges.size == 0:
            self._publish(0.0, 0.0)
            return

        self._extend_disparities(ranges, msg.angle_increment)

        bias_rad, free_l, free_r = self._corridor_bias(ranges, angles)

        best = self._select_heading(ranges, angles)
        target_rad = float(angles[best]) + bias_rad
        steer_norm = self._steering_command(target_rad)

        depth_m = self._path_depth(ranges, angles, steer_norm)
        throttle = self._throttle_for(depth_m, steer_norm)

        self._log_state(msg, target_rad, depth_m, steer_norm, throttle,
                        bias_rad, free_l, free_r)
        self._publish(steer_norm, throttle)

    def _prepare(self, msg):
        """
        Return bumper-referenced ranges and their bearings across the forward
        field of view. Non-returns arrive as +inf and are treated as the
        horizon, since open space is exactly what they represent.
        """
        r = np.asarray(msg.ranges, dtype=np.float64)
        r = np.nan_to_num(r, nan=0.0, posinf=self.horizon_m, neginf=0.0)
        r = np.maximum(r - LIDAR_TO_BUMPER_M, 0.0)
        r = np.minimum(r, self.horizon_m)

        angles = msg.angle_min + np.arange(r.size) * msg.angle_increment
        keep = np.abs(angles) <= self.fov_rad
        return r[keep], angles[keep]

    def _extend_disparities(self, ranges, angle_increment):
        """
        Overwrite samples adjacent to each obstacle edge with the nearer
        reading, across the angular span half a car width subtends at that
        range.

        For a disparity between samples i and i+1, the nearer of the two marks
        an edge the vehicle must clear. The number of samples to overwrite is

            n = arctan(half_width / d) / angle_increment

        where d is the nearer range. Overwriting proceeds away from the nearer
        sample. Using np.minimum rather than assignment means overlapping
        extensions from several disparities compose correctly instead of the
        last one written winning.
        """
        half_car = CAR_HALF_WIDTH_M + self.extend_margin_m
        diffs = np.diff(ranges)

        for i in np.flatnonzero(np.abs(diffs) > self.disparity_threshold_m):
            if diffs[i] > 0:
                near_idx, near_d, step = i, ranges[i], 1
            else:
                near_idx, near_d, step = i + 1, ranges[i + 1], -1

            if near_d < 1e-3:
                continue

            n = int(math.atan2(half_car, near_d) / angle_increment) + 1

            if step > 0:
                lo, hi = near_idx, min(near_idx + n + 1, ranges.size)
            else:
                lo, hi = max(near_idx - n, 0), near_idx + 1

            ranges[lo:hi] = np.minimum(ranges[lo:hi], near_d)

    def _select_heading(self, ranges, angles):
        """
        Return the index of the chosen heading.

        Taking a plain argmax lets the aim settle on whichever sample happens
        to read marginally longest, which in practice is often at the edge of
        the window: the corridor genuinely opens sideways in a corner, and a
        boundary sample can never be disparity-extended because np.diff has no
        neighbour for it. Collecting all samples within a tolerance of the
        maximum and choosing the most forward of them keeps the aim ahead
        whenever several headings are effectively equally open.
        """
        if self.tie_tolerance_m <= 0.0:
            return int(np.argmax(ranges))

        peak = float(np.max(ranges))
        near_max = np.flatnonzero(ranges >= peak - self.tie_tolerance_m)
        return int(near_max[np.argmin(np.abs(angles[near_max]))])

    def _steering_command(self, target_rad):
        """
        Convert an aim bearing to a normalised steering command.

        Pure pursuit gives delta = atan(2 * L * sin(alpha) / Ld), which cannot
        demand more lock than the geometry justifies. A 60 deg bearing at a
        1.5 m lookahead yields 0.35 rad rather than the 1.05 rad a linear
        mapping would ask for, while small bearings remain close to linear.
        """
        if self.use_pure_pursuit:
            delta = math.atan2(
                2.0 * WHEELBASE_M * math.sin(target_rad), self.lookahead_m)
        else:
            delta = self.steering_gain * target_rad

        return float(np.clip(delta / MAX_STEER_RAD, -1.0, 1.0))

    def _front_depth(self, ranges, angles):
        """
        Minimum free distance inside a narrow cone straight ahead.

        Depth at the aim index is the wrong input for a speed decision. Part
        way through a corner the aim points at open track 40 to 58 degrees off
        axis, so that reading saturates near the sensing horizon while the
        vehicle is still turning, and the throttle ramp responds as though the
        road were straight. Logged telemetry showed 0.21 throttle at an aim
        bearing of 58 degrees, which is power on mid-corner and produced
        repeated exit collisions.

        Taking the minimum over a narrow forward cone answers the question the
        speed controller actually needs: how far can this vehicle travel along
        its current heading before the corridor constrains it.
        """
        mask = np.abs(angles) <= self.front_cone_rad
        if not mask.any():
            return float(np.min(ranges))
        return float(np.min(ranges[mask]))

    def _corridor_bias(self, ranges, angles):
        """
        Measure the lateral asymmetry of the corridor and derive a steering
        bias from it. DIAGNOSTIC ONLY at bias_gain 0.0.

        A gap follower that aims at the deepest visible point traces a path
        near the centre of the corridor. That is not a racing line: a racing
        line runs wide on entry, tightens to an apex, and opens out again on
        exit, so that the vehicle is close to straight for much of the corner.
        Logged telemetry shows the present aim bearing sitting between 0.4 and
        0.9 rad almost continuously, meaning the vehicle is steering, and
        therefore derating throttle, essentially all the time.

        The corner geometry cannot be known ahead of time from an 8 m scan, but
        its asymmetry can be measured. Free distance is averaged over a window
        centred on +bias_side_deg (left) and again on -bias_side_deg (right).
        Approaching a right hand corner the left side is the more open of the
        two, at the apex the two are comparable and both short, and on exit the
        right opens up again. The normalised difference

            bias = (free_left - free_right) / (free_left + free_right)

        is therefore positive when the open space lies to the left, negative
        when it lies to the right, and near zero on a straight or at an apex.
        Adding a multiple of it to the aim bearing pushes the vehicle toward
        the outside of the corner on entry and lets it run out again on exit.

        Returns the bias in radians together with the two raw averages, so the
        measure can be validated against observed behaviour before it is given
        any authority over steering.
        """
        half = self.bias_window_rad

        left = np.abs(angles - self.bias_side_rad) <= half
        right = np.abs(angles + self.bias_side_rad) <= half

        free_l = float(np.mean(ranges[left])) if left.any() else 0.0
        free_r = float(np.mean(ranges[right])) if right.any() else 0.0

        total = free_l + free_r
        if total < 1e-3:
            return 0.0, free_l, free_r

        normalised = (free_l - free_r) / total
        return self.bias_gain * normalised, free_l, free_r

    def _path_depth_v1(self, ranges, angles, steer_norm):
        """
        Free distance measured along the arc the vehicle is actually following.

        Two earlier measures were tried and both are wrong in opposite ways.

        Taking the range at the aim index is too optimistic: part way through a
        corner the aim points at open track 40 to 58 degrees off axis, so the
        reading saturates near the sensing horizon while the vehicle is still
        turning. Logged telemetry showed 0.21 throttle at an aim bearing of
        58 degrees, which is power on mid-corner, and produced repeated exit
        collisions.

        Taking the minimum over a cone fixed to the chassis is too pessimistic:
        a cone locked to the current heading points straight at the outside
        barrier through a corner, so depth collapsed below 0.5 m for much of
        the lap and throttle pinned at its floor. Lap time regressed from 8.8 s
        to 12.0 s.

        The correct question is how far this vehicle can travel before the
        corridor constrains it, and that depends on the path, not the heading.
        The steering angle implies a turn radius R = L / tan(delta), so the
        vehicle is following a circle of that radius centred abeam. A scan
        point at range r and bearing theta lies at (r cos theta, r sin theta);
        its distance from the arc centre determines whether the vehicle will
        pass through it. Points whose radial offset from the arc exceeds half
        the swept path width are missed and are ignored. Of the remainder, the
        nearest measured along the arc is the true constraint.

        As delta tends to zero the radius diverges and the test degenerates to
        a straight corridor of the same width, which is the correct limiting
        behaviour.
        """
        delta = steer_norm * MAX_STEER_RAD
        tan_delta = math.tan(delta)

        xs = ranges * np.cos(angles)
        ys = ranges * np.sin(angles)

        if abs(tan_delta) < 1e-3:
            # Straight ahead: a rectangular corridor of the swept width.
            on_path = (np.abs(ys) <= self.path_half_width_m) & (xs > 0.0)
            if not on_path.any():
                return self.arc_max_m
            return float(np.clip(np.min(xs[on_path]), 0.0, self.arc_max_m))

        radius = WHEELBASE_M / tan_delta          # signed; left turn positive
        centre_y = radius

        # Radial distance of each point from the arc centre.
        offset = np.abs(np.hypot(xs, ys - centre_y) - abs(radius))
        on_path = offset <= self.path_half_width_m

        # Angle subtended from the centre, measured forward along the arc.
        phi = np.arctan2(xs, np.sign(radius) * (centre_y - ys))
        forward = phi > 0.0

        valid = on_path & forward
        if not valid.any():
            return self.arc_max_m

        arc_len = abs(radius) * phi[valid]
        return float(np.clip(np.min(arc_len), 0.0, self.arc_max_m))

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

    def _throttle_for(self, depth_m, steer_norm):
        """
        Scale throttle with the free distance at the chosen heading.

        After disparity extension the depth at the aim index measures how far
        the vehicle can travel before the corridor constrains it: on a straight
        it saturates at the sensing horizon, and entering a corner it
        collapses. Ramping throttle linearly between two depth thresholds
        therefore brakes on corner approach and accelerates on exit, using
        nothing but the current scan.

        This is deliberately open loop. An earlier revision derived a speed
        target from the steering angle and closed a loop on measured speed,
        which coupled speed to lookahead to aim bearing to curvature and back
        to speed; logged throttle showed a self-sustaining oscillation of about
        0.8 s that no amount of filtering or rate limiting removed. Here the
        command depends only on the scan, so there is no path from the
        vehicle's own motion back into its own command and the loop cannot
        close.

        Negative throttle is never emitted because it engages reverse rather
        than braking on this vehicle. Deceleration is by idle torque alone,
        which is why the ramp must begin well before the corner: depth_min_m
        and depth_max_m should be set so that throttle is already falling
        while the corner is still several metres away.
        """
        if self.use_depth_throttle:
            span = max(self.depth_max_m - self.depth_min_m, 1e-3)
            frac = (depth_m - self.depth_min_m) / span
            frac = float(np.clip(frac, 0.0, 1.0))
            throttle = self.throttle_min + frac * (
                self.throttle_max - self.throttle_min)
        else:
            throttle = self.throttle

        # Cut power with steering angle, raised to an exponent.
        #
        # A linear derate penalises every correction equally, so the small
        # inputs used to hold a straight line cost real speed: logged telemetry
        # showed a commanded 0.22 delivered as 0.179 at only 0.10 rad of steer.
        # Raising the normalised steering to a power leaves those corrections
        # nearly untouched while preserving the full cut at lock. At exponent
        # 2.0 a tenth of lock costs one per cent instead of ten, and full lock
        # still yields zero throttle.
        #
        # The cut is needed at all because the lateral tire curve peaks at 0.01
        # slip and has halved by 0.10, so applying drive torque while cornering
        # hard pushes the vehicle past the grip peak and it runs wide on exit.
        cut = self.steer_derate * (abs(steer_norm) ** self.derate_exponent)
        throttle *= max(1.0 - cut, 0.0)
        return max(throttle, 0.0)

    def _log_state(self, msg, target_rad, depth_m, steer_norm, throttle,
                   bias_rad=0.0, free_l=0.0, free_r=0.0):
        """Emit a one-line state summary about once per second."""
        now_s = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if now_s - self.last_log_s < self.log_period_s:
            return
        self.last_log_s = now_s
        self.get_logger().info(
            'aim={:+.3f}  steer={:+.3f}  thr={:.3f}  '
            'd1={:.2f} d2={:.2f} dc={:.2f} da={:.2f}  '
            'L={:.2f} R={:.2f} bias={:+.3f}'.format(
                target_rad, steer_norm, throttle,
                depth_m, self.depth_v2, self.depth_cone,
                self.depth_aim,
                free_l, free_r, bias_rad))

    def _publish(self, steer_norm, throttle):
        steer_msg = Float32()
        steer_msg.data = float(np.clip(steer_norm, -1.0, 1.0))
        self.steer_pub.publish(steer_msg)

        throttle_msg = Float32()
        throttle_msg.data = float(np.clip(throttle, 0.0, 1.0))
        self.throttle_pub.publish(throttle_msg)


def main(args=None):
    rclpy.init(args=args)
    node = GapFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            stop = Float32()
            stop.data = 0.0
            node.throttle_pub.publish(stop)
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
