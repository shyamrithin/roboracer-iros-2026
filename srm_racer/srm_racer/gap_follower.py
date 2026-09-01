#!/usr/bin/env python3
# =============================================================================
# File        : gap_follower.py
# Package     : srm_racer
# Project     : RoboRacer Sim Racing League @ IROS 2026 (AutoDRIVE Ecosystem)
# Target      : ROS 2 Humble / Python 3.10
# Created     : 2026-08-31
# =============================================================================
# CODE DESCRIPTION
# -----------------------------------------------------------------------------
# Reactive "follow the gap" controller for the RoboRacer digital twin in the
# AutoDRIVE Simulator. Consumes only competition-legal input streams (2D LiDAR)
# and emits normalised steering and throttle commands. No map, no memory, no
# ground-truth pose, so behaviour transfers to an unseen racetrack.
#
# Pipeline, once per incoming laser scan:
#   1. Sanitise ranges (NaN and +inf are replaced, values clipped to a horizon).
#   2. Subtract the LiDAR-to-bumper offset so ranges describe bumper clearance.
#   3. Restrict attention to a forward field of view.
#   4. Zero out a "safety bubble" around the nearest obstacle.
#   5. Extend disparities by half the car width so no gap narrower than the
#      vehicle is ever selected (guards against the gaps between track ducts).
#   6. Select the widest remaining gap, reject it if physically too narrow,
#      and aim at its angular centre.
#   7. Rate-limit the steering command and scale throttle by forward clearance.
#
# IMPORTANT DESIGN CONSTRAINTS
#   * Every rate limit is expressed per second and multiplied by the measured
#     dt between scans. The observed bridge tick on the development laptop is
#     ~18.5 Hz while the documented sensor rate is 40 Hz; the evaluation
#     workstation will likely differ again. Time-based logic keeps behaviour
#     identical across all three.
#   * Negative throttle engages REVERSE on this vehicle, it does not brake.
#     Deceleration is achieved by commanding zero throttle and letting the
#     simulated idle torque act as a braking torque. Throttle is therefore
#     clamped to be non-negative.
#   * Restricted topics (ips, odom, tf, lap/collision telemetry, reset_command)
#     are deliberately never subscribed to, per the competition rule book.
#
# GEOMETRY CONSTANTS (from the 2026 Technical Guide)
#   Car width 0.270 m, wheelbase 0.324 m, front overhang 0.090 m.
#   LiDAR frame at x = 0.2733 m from the rear axle; front bumper at 0.414 m.
#   Steering angle limits +/- 0.5236 rad, mapped to a normalised [-1, 1] command.
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
MAX_STEER_RAD = 0.5236         # +/- 30 deg mechanical limit


class GapFollower(Node):
    """Reactive gap-following racing controller."""

    def __init__(self):
        super().__init__('gap_follower')

        # --- Tunable parameters ---------------------------------------------
        self.declare_parameter('fov_deg', 90.0)
        self.declare_parameter('horizon_m', 6.0)
        self.declare_parameter('bubble_radius_m', 0.30)
        self.declare_parameter('clearance_margin_m', 0.055)
        self.declare_parameter('gap_threshold_m', 1.0)
        self.declare_parameter('min_gap_width_m', 0.45)
        self.declare_parameter('steer_rate_rad_s', 2.5)
        self.declare_parameter('throttle_cruise', 0.06)
        self.declare_parameter('throttle_min', 0.02)
        self.declare_parameter('clearance_full_m', 3.0)
        self.declare_parameter('clearance_stop_m', 0.5)
        self.declare_parameter('front_cone_deg', 12.0)

        self._reload_parameters()

        # --- State ------------------------------------------------------------
        self.prev_stamp_s = None
        self.prev_steer_rad = 0.0

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

        self.get_logger().info('gap_follower ready, waiting for laser scans')

    # -------------------------------------------------------------------------
    def _reload_parameters(self):
        """Pull current parameter values into plain attributes."""
        g = self.get_parameter
        self.fov_rad = math.radians(g('fov_deg').value)
        self.horizon_m = g('horizon_m').value
        self.bubble_radius_m = g('bubble_radius_m').value
        self.clearance_margin_m = g('clearance_margin_m').value
        self.gap_threshold_m = g('gap_threshold_m').value
        self.min_gap_width_m = g('min_gap_width_m').value
        self.steer_rate_rad_s = g('steer_rate_rad_s').value
        self.throttle_cruise = g('throttle_cruise').value
        self.throttle_min = g('throttle_min').value
        self.clearance_full_m = g('clearance_full_m').value
        self.clearance_stop_m = g('clearance_stop_m').value
        self.front_cone_rad = math.radians(g('front_cone_deg').value)

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

        start, end = self._widest_gap(window)

        if start is None:
            # Fully enclosed. Hold the previous heading and stop driving.
            self.get_logger().warn('no viable gap found, coasting',
                                   throttle_duration_sec=1.0)
            self._publish(self.prev_steer_rad / MAX_STEER_RAD, 0.0)
            return

        centre_index = lo + (start + end) // 2
        target_rad = msg.angle_min + centre_index * msg.angle_increment
        steer_rad = self._rate_limit(target_rad, dt)

        clearance_m = self._forward_clearance(ranges, msg)
        throttle = self._throttle_for(clearance_m)

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
        # Guard against duplicate stamps and long stalls.
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
        sideways by half the car width plus margin. This prevents selecting a
        gap the car cannot physically fit through, including apparent openings
        between the cylindrical track ducts.
        """
        half_car = CAR_HALF_WIDTH_M + self.clearance_margin_m
        diffs = np.diff(window)
        threshold = 0.35

        for i in np.flatnonzero(np.abs(diffs) > threshold):
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

    def _widest_gap(self, window):
        """
        Return (start, end) indices of the widest run of free space that is
        also physically wide enough for the car. (None, None) if none qualifies.
        """
        free = window > self.gap_threshold_m
        if not free.any():
            return None, None

        best = (None, None, 0)
        run_start = None

        for i, is_free in enumerate(free):
            if is_free and run_start is None:
                run_start = i
            elif not is_free and run_start is not None:
                length = i - run_start
                if length > best[2]:
                    best = (run_start, i, length)
                run_start = None

        if run_start is not None:
            length = free.size - run_start
            if length > best[2]:
                best = (run_start, free.size, length)

        start, end, _ = best
        if start is None:
            return None, None

        # Reject gaps that are angularly wide but physically narrow.
        depth = float(np.mean(window[start:end]))
        angular_width = (end - start) * (2 * self.fov_rad / window.size)
        physical_width = 2.0 * depth * math.tan(angular_width / 2.0)
        if physical_width < self.min_gap_width_m:
            return None, None

        return start, end

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

    def _throttle_for(self, clearance_m):
        """
        Map forward clearance to a non-negative throttle. Negative throttle is
        never emitted because it engages reverse rather than braking.
        """
        if clearance_m <= self.clearance_stop_m:
            return 0.0
        span = max(self.clearance_full_m - self.clearance_stop_m, 1e-3)
        frac = (clearance_m - self.clearance_stop_m) / span
        frac = float(np.clip(frac, 0.0, 1.0))
        throttle = self.throttle_min + frac * (self.throttle_cruise - self.throttle_min)
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
        stop = Float32()
        stop.data = 0.0
        node.throttle_pub.publish(stop)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
