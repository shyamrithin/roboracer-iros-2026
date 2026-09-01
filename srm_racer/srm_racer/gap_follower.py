#!/usr/bin/env python3
# =============================================================================
# File        : gap_follower.py
# Package     : srm_racer
# Project     : RoboRacer Sim Racing League @ IROS 2026 (AutoDRIVE Ecosystem)
# Target      : ROS 2 Humble / Python 3.10
# Created     : 2026-08-31
# Revised     : 2026-09-01  (v2: pure pursuit steering, gap hysteresis)
# =============================================================================
# CODE DESCRIPTION
# -----------------------------------------------------------------------------
# Reactive "follow the gap" controller for the RoboRacer digital twin in the
# AutoDRIVE Simulator. Consumes only competition-legal input streams (2D LiDAR)
# and emits normalised steering and throttle commands. No map, no memory of the
# track, and no ground-truth pose, so behaviour transfers to an unseen circuit.
#
# REVISION NOTES (v2)
# -----------------------------------------------------------------------------
# v1 mapped the bearing of the gap centre directly onto the steering command.
# That is dimensionally wrong: a gap 60 deg off the nose demanded 60 deg of
# steer against a 30 deg mechanical limit, so the command clipped to full lock
# and stayed there. Logged telemetry showed runs of up to 18 consecutive
# samples at -1.0, with transitions advancing by exactly the slew limit each
# tick, producing a full-lock-to-full-lock limit cycle in every corner.
#
# v2 corrects this in two places:
#   1. Steering is now computed by the pure pursuit geometric law,
#          delta = atan(2 * L * sin(alpha) / Ld)
#      where L is the wheelbase, alpha the bearing to the aim point and Ld the
#      lookahead distance. Lookahead scales with forward clearance, so the car
#      looks further ahead on straights and closer in tight corners.
#   2. Gap selection is scored rather than simply "widest wins". The score
#      rewards depth and physical width but penalises angular distance from the
#      previously chosen heading. This hysteresis stops the aim point flipping
#      between two similar candidates on alternate scans, which is what excited
#      the oscillation.
#
# PIPELINE, once per incoming laser scan:
#   1. Sanitise ranges (NaN and +inf replaced, values clipped to a horizon).
#   2. Subtract the LiDAR-to-bumper offset so ranges describe bumper clearance.
#   3. Restrict attention to a forward field of view.
#   4. Zero a safety bubble around the nearest return.
#   5. Extend disparities by half the car width so no gap narrower than the
#      vehicle is ever selected (guards the gaps between track ducts).
#   6. Score all candidate gaps, pick the best, take its centre as aim bearing.
#   7. Convert that bearing to a steering angle via pure pursuit, slew-limit it,
#      and scale throttle by forward clearance.
#
# DESIGN CONSTRAINTS
#   * Every rate limit is expressed per second and multiplied by the measured
#     dt between scans. The bridge tick observed on the development laptop is
#     ~18.5 Hz against a documented 40 Hz sensor rate; the evaluation
#     workstation will differ again. Time-based logic keeps behaviour identical
#     across all of them.
#   * Negative throttle engages REVERSE on this vehicle rather than braking, so
#     throttle is clamped non-negative and deceleration relies on the simulated
#     idle braking torque.
#   * Restricted topics (ips, odom, tf, lap and collision telemetry,
#     reset_command) are never subscribed to, per section 2.4 of the rule book.
#
# GEOMETRY (2026 Technical Guide)
#   Car width 0.270 m, wheelbase 0.324 m, front overhang 0.090 m.
#   LiDAR frame at x = 0.2733 m from the rear axle; front bumper at 0.414 m.
#   Steering limits +/- 0.5236 rad, actuator slew 3.2 rad/s.
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
    """Reactive gap-following racing controller with pure pursuit steering."""

    def __init__(self):
        super().__init__('gap_follower')

        # --- Perception parameters -------------------------------------------
        self.declare_parameter('fov_deg', 90.0)
        self.declare_parameter('horizon_m', 6.0)
        self.declare_parameter('bubble_radius_m', 0.30)
        self.declare_parameter('clearance_margin_m', 0.055)
        self.declare_parameter('gap_threshold_m', 1.0)
        self.declare_parameter('min_gap_width_m', 0.45)
        self.declare_parameter('disparity_threshold_m', 0.35)

        # --- Gap scoring weights ---------------------------------------------
        self.declare_parameter('w_depth', 1.0)
        self.declare_parameter('w_width', 0.5)
        self.declare_parameter('w_hysteresis', 1.5)

        # --- Steering parameters ---------------------------------------------
        self.declare_parameter('lookahead_gain', 0.6)
        self.declare_parameter('lookahead_min_m', 0.8)
        self.declare_parameter('lookahead_max_m', 3.0)
        self.declare_parameter('steer_rate_rad_s', 3.0)

        # --- Throttle parameters ---------------------------------------------
        self.declare_parameter('throttle_cruise', 0.12)
        self.declare_parameter('throttle_min', 0.03)
        self.declare_parameter('clearance_full_m', 3.0)
        self.declare_parameter('clearance_stop_m', 0.5)
        self.declare_parameter('front_cone_deg', 12.0)
        self.declare_parameter('steer_throttle_derate', 0.5)

        self._reload_parameters()

        # --- State ------------------------------------------------------------
        self.prev_stamp_s = None
        self.prev_steer_rad = 0.0
        self.prev_bearing_rad = 0.0

        # --- ROS interfaces ---------------------------------------------------
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

        self.get_logger().info('gap_follower v2 ready, waiting for laser scans')

    # -------------------------------------------------------------------------
    def _reload_parameters(self):
        """Pull current parameter values into plain attributes each cycle."""
        g = self.get_parameter
        self.fov_rad = math.radians(g('fov_deg').value)
        self.horizon_m = g('horizon_m').value
        self.bubble_radius_m = g('bubble_radius_m').value
        self.clearance_margin_m = g('clearance_margin_m').value
        self.gap_threshold_m = g('gap_threshold_m').value
        self.min_gap_width_m = g('min_gap_width_m').value
        self.disparity_threshold_m = g('disparity_threshold_m').value

        self.w_depth = g('w_depth').value
        self.w_width = g('w_width').value
        self.w_hysteresis = g('w_hysteresis').value

        self.lookahead_gain = g('lookahead_gain').value
        self.lookahead_min_m = g('lookahead_min_m').value
        self.lookahead_max_m = g('lookahead_max_m').value
        self.steer_rate_rad_s = g('steer_rate_rad_s').value

        self.throttle_cruise = g('throttle_cruise').value
        self.throttle_min = g('throttle_min').value
        self.clearance_full_m = g('clearance_full_m').value
        self.clearance_stop_m = g('clearance_stop_m').value
        self.front_cone_rad = math.radians(g('front_cone_deg').value)
        self.steer_throttle_derate = g('steer_throttle_derate').value

    # -------------------------------------------------------------------------
    def scan_callback(self, msg: LaserScan):
        """Main control loop, executed once per laser scan."""
        self._reload_parameters()

        dt = self._elapsed_time(msg)
        if dt is None:
            return

        ranges = self._sanitise(msg)
        lo, hi = self._fov_slice(msg)
        window = ranges[lo:hi].copy()

        if window.size == 0:
            self._publish(0.0, 0.0)
            return

        self._apply_safety_bubble(window, msg.angle_increment)
        self._extend_disparities(window, msg.angle_increment)

        clearance_m = self._forward_clearance(ranges, msg)
        bearing_rad = self._select_gap(window, msg, lo)

        if bearing_rad is None:
            # Fully enclosed. Hold the last heading and stop driving.
            self.get_logger().warn('no viable gap found, coasting',
                                   throttle_duration_sec=1.0)
            self._publish(self.prev_steer_rad / MAX_STEER_RAD, 0.0)
            return

        self.prev_bearing_rad = bearing_rad

        lookahead_m = float(np.clip(
            clearance_m * self.lookahead_gain,
            self.lookahead_min_m,
            self.lookahead_max_m,
        ))

        target_steer_rad = self._pure_pursuit(bearing_rad, lookahead_m)
        steer_rad = self._rate_limit(target_steer_rad, dt)
        throttle = self._throttle_for(clearance_m, steer_rad)

        self._publish(steer_rad / MAX_STEER_RAD, throttle)

    # -------------------------------------------------------------------------
    def _elapsed_time(self, msg):
        """Seconds since the previous scan, or None on the very first scan."""
        stamp_s = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.prev_stamp_s is None:
            self.prev_stamp_s = stamp_s
            return None
        dt = stamp_s - self.prev_stamp_s
        self.prev_stamp_s = stamp_s
        return float(np.clip(dt, 0.005, 0.25))

    def _sanitise(self, msg):
        """Return bumper-referenced ranges with NaN and inf removed."""
        r = np.asarray(msg.ranges, dtype=np.float64)
        r = np.nan_to_num(r, nan=0.0, posinf=msg.range_max, neginf=0.0)
        r = np.maximum(r - LIDAR_TO_BUMPER_M, 0.0)
        return np.minimum(r, self.horizon_m)

    def _fov_slice(self, msg):
        """Index bounds of the forward field of view."""
        lo = int((-self.fov_rad - msg.angle_min) / msg.angle_increment)
        hi = int((self.fov_rad - msg.angle_min) / msg.angle_increment)
        n = len(msg.ranges)
        return max(lo, 0), min(hi, n)

    def _apply_safety_bubble(self, window, angle_increment):
        """Zero a bubble around the nearest return so we never aim at it."""
        nearest = int(np.argmin(window))
        d = max(window[nearest], 1e-3)
        half_span = int(math.atan2(self.bubble_radius_m, d) / angle_increment)
        lo = max(nearest - half_span, 0)
        hi = min(nearest + half_span + 1, window.size)
        window[lo:hi] = 0.0

    def _extend_disparities(self, window, angle_increment):
        """
        Where consecutive returns differ sharply, project the nearer surface
        sideways by half the car width plus margin, so that a gap the vehicle
        cannot physically fit through is never a candidate. This also closes
        the apparent openings between the cylindrical track ducts.
        """
        half_car = CAR_HALF_WIDTH_M + self.clearance_margin_m
        diffs = np.diff(window)

        for i in np.flatnonzero(np.abs(diffs) > self.disparity_threshold_m):
            if diffs[i] > 0:
                near_idx, near_d, direction = i, window[i], 1
            else:
                near_idx, near_d, direction = i + 1, window[i + 1], -1

            if near_d < 1e-3:
                continue

            span = int(math.atan2(half_car, near_d) / angle_increment)
            if direction > 0:
                lo, hi = near_idx, min(near_idx + span + 1, window.size)
            else:
                lo, hi = max(near_idx - span, 0), near_idx + 1
            window[lo:hi] = np.minimum(window[lo:hi], near_d)

    def _select_gap(self, window, msg, lo):
        """
        Score every viable gap and return the aim bearing of the best one.

        The score rewards depth and physical width, and penalises angular
        distance from the bearing chosen on the previous scan. Without that
        penalty the choice flips between similar candidates on alternate
        scans, which drives the steering into a full-lock limit cycle.
        """
        free = window > self.gap_threshold_m
        if not free.any():
            return None

        runs = []
        run_start = None
        for i, is_free in enumerate(free):
            if is_free and run_start is None:
                run_start = i
            elif not is_free and run_start is not None:
                runs.append((run_start, i))
                run_start = None
        if run_start is not None:
            runs.append((run_start, free.size))

        best_score = -math.inf
        best_bearing = None

        for start, end in runs:
            depth = float(np.mean(window[start:end]))
            angular_width = (end - start) * msg.angle_increment
            physical_width = 2.0 * depth * math.tan(angular_width / 2.0)

            if physical_width < self.min_gap_width_m:
                continue

            centre_index = lo + (start + end) // 2
            bearing = msg.angle_min + centre_index * msg.angle_increment

            score = (
                self.w_depth * depth
                + self.w_width * physical_width
                - self.w_hysteresis * abs(bearing - self.prev_bearing_rad)
            )

            if score > best_score:
                best_score = score
                best_bearing = bearing

        return best_bearing

    def _pure_pursuit(self, bearing_rad, lookahead_m):
        """
        Geometric steering law for an Ackermann vehicle chasing an aim point
        at bearing `bearing_rad` and distance `lookahead_m`:

            delta = atan(2 * L * sin(alpha) / Ld)

        Unlike a direct bearing-to-steer mapping this cannot demand more lock
        than the geometry justifies, so wide aim bearings no longer saturate
        the actuator.
        """
        delta = math.atan2(2.0 * WHEELBASE_M * math.sin(bearing_rad), lookahead_m)
        return float(np.clip(delta, -MAX_STEER_RAD, MAX_STEER_RAD))

    def _rate_limit(self, target_rad, dt):
        """Slew-limit the steering target and clamp to the mechanical limit."""
        max_step = self.steer_rate_rad_s * dt
        delta = float(np.clip(target_rad - self.prev_steer_rad, -max_step, max_step))
        steer = float(np.clip(self.prev_steer_rad + delta, -MAX_STEER_RAD, MAX_STEER_RAD))
        self.prev_steer_rad = steer
        return steer

    def _forward_clearance(self, ranges, msg):
        """Minimum bumper clearance inside a narrow cone dead ahead."""
        lo = int((-self.front_cone_rad - msg.angle_min) / msg.angle_increment)
        hi = int((self.front_cone_rad - msg.angle_min) / msg.angle_increment)
        lo, hi = max(lo, 0), min(hi, ranges.size)
        if hi <= lo:
            return self.horizon_m
        return float(np.min(ranges[lo:hi]))

    def _throttle_for(self, clearance_m, steer_rad):
        """
        Map forward clearance to a non-negative throttle, then derate for
        steering angle so the car slows into corners. Negative throttle is
        never emitted because it engages reverse rather than braking.
        """
        if clearance_m <= self.clearance_stop_m:
            return 0.0

        span = max(self.clearance_full_m - self.clearance_stop_m, 1e-3)
        frac = float(np.clip((clearance_m - self.clearance_stop_m) / span, 0.0, 1.0))
        throttle = self.throttle_min + frac * (self.throttle_cruise - self.throttle_min)

        steer_frac = abs(steer_rad) / MAX_STEER_RAD
        throttle *= (1.0 - self.steer_throttle_derate * steer_frac)

        return max(throttle, 0.0)

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
        # Best effort: leave the vehicle stationary before the context closes.
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
