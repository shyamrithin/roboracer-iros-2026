#!/usr/bin/env python3
# =============================================================================
# File        : gap_follower.py
# Package     : srm_racer
# Project     : RoboRacer Sim Racing League @ IROS 2026 (AutoDRIVE Ecosystem)
# Target      : ROS 2 Humble / Python 3.10
# Created     : 2026-08-31
# Revised     : 2026-09-01  (v2: pure pursuit steering, gap hysteresis)
# Revised     : 2026-09-02  (v3: encoder speed estimate, speed-scaled lookahead,
#                            closed-loop speed control)
# =============================================================================
# CODE DESCRIPTION
# -----------------------------------------------------------------------------
# Reactive "follow the gap" controller for the RoboRacer digital twin in the
# AutoDRIVE Simulator. Consumes only competition-legal input streams (2D LiDAR
# and wheel encoders) and emits normalised steering and throttle commands. No
# map, no stored knowledge of the circuit, and no ground-truth pose, so the
# behaviour transfers to an unseen racetrack.
#
# REVISION NOTES (v3)
# -----------------------------------------------------------------------------
# v2 scaled the pure pursuit lookahead distance from forward LiDAR clearance.
# That is backwards. Clearance collapses inside a corner, so the lookahead fell
# to its 0.8 m floor exactly where the aim bearing was widest. At that floor the
# steering law reduces to delta = atan(0.81 * sin(alpha)), which saturates the
# 0.5236 rad mechanical limit for any bearing beyond roughly 45 deg. Logged
# telemetry at throttle 0.20 showed runs of 13 to 14 consecutive samples pinned
# at -1.0, once per corner.
#
# v2 also derated throttle directly from instantaneous steering angle. That
# closed a positive feedback path: steer hard, cut throttle, vehicle slows,
# clearance and gap geometry shift, steering changes, throttle jumps. Logged
# throttle chattered across the full commanded range within a few samples,
# pitching the suspension continuously.
#
# v3 addresses both:
#   1. Vehicle speed is estimated by differentiating the wheel encoder
#      positions, which are permissible inputs at run time. Lookahead is then
#      Ld = base + gain * v, the conventional pure pursuit formulation, with a
#      floor of 1.2 m. That floor is chosen so that
#          atan(2 * L * sin(alpha) / 1.2) < 0.5236 rad  for all alpha,
#      making steering saturation geometrically impossible.
#   2. Throttle now closes a loop on a speed target rather than reacting to
#      instantaneous geometry. The target is the minimum of a curvature limit
#      (from the lateral acceleration budget), a clearance limit (from the
#      available stopping distance), and an absolute cap. A feedforward term
#      plus proportional correction produces the command, and the result is
#      low-pass filtered. Throttle becomes a smooth function of speed error
#      instead of a fast function of scan geometry.
#
# PIPELINE, once per incoming laser scan:
#   1. Sanitise ranges (NaN and +inf replaced, values clipped to a horizon).
#   2. Subtract the LiDAR-to-bumper offset so ranges describe bumper clearance.
#   3. Restrict attention to a forward field of view.
#   4. Zero a safety bubble around the nearest return.
#   5. Extend disparities by half the car width so no gap narrower than the
#      vehicle is ever selected (this also closes the apparent openings between
#      the cylindrical track ducts).
#   6. Score all candidate gaps and pick the best; its centre is the aim
#      bearing. Scoring penalises angular distance from the previous choice so
#      the aim point cannot flip between similar candidates on alternate scans.
#   7. Convert the bearing to a steering angle by pure pursuit with a
#      speed-scaled lookahead, then slew-limit it.
#   8. Derive a speed target, close the loop on measured speed, filter, publish.
#
# DESIGN CONSTRAINTS
#   * Every rate limit and filter is expressed per second and multiplied by the
#     measured dt between scans. The bridge tick observed on the development
#     laptop is ~18.5 Hz against a documented 40 Hz sensor rate, and the
#     evaluation workstation will differ again. Time-based logic keeps
#     behaviour identical across all of them.
#   * Negative throttle engages REVERSE on this vehicle rather than braking, so
#     throttle is clamped non-negative and deceleration relies on the simulated
#     idle braking torque.
#   * Restricted topics (ips, odom, tf, lap and collision telemetry, and
#     reset_command) are never subscribed to, per section 2.4 of the rule book.
#     Wheel encoders are explicitly permissible inputs.
#
# ENCODER SCALING
#   sensor_msgs/JointState.position is populated as a float and velocity is
#   empty, so speed is obtained by differentiation. The position field is taken
#   to be wheel angle in radians, giving v = r * dtheta/dt with r = 0.0590 m.
#   If the field is in fact raw ticks, the correct scale is instead
#   0.3707 m circumference / 1920 ticks per revolution = 1.931e-4 m per tick,
#   a factor of 305.6 smaller. The scale is exposed as the parameter
#   `encoder_m_per_unit` and the estimate is logged once per second so it can
#   be checked against the simulator HUD speed readout.
#
# GEOMETRY (2026 Technical Guide)
#   Car width 0.270 m, wheelbase 0.324 m, front overhang 0.090 m.
#   Wheel radius 0.0590 m, 16 pulses per revolution, conversion ratio 120.
#   LiDAR frame at x = 0.2733 m from the rear axle; front bumper at 0.414 m.
#   Steering limits +/- 0.5236 rad, actuator slew 3.2 rad/s.
# =============================================================================

import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan, JointState
from std_msgs.msg import Float32

# --- Fixed vehicle geometry, do not tune -------------------------------------
LIDAR_TO_BUMPER_M = 0.141      # 0.414 m bumper - 0.2733 m LiDAR mount
CAR_HALF_WIDTH_M = 0.135       # 0.270 m overall width
WHEELBASE_M = 0.324
WHEEL_RADIUS_M = 0.0590
MAX_STEER_RAD = 0.5236         # +/- 30 deg mechanical limit


class GapFollower(Node):
    """Reactive gap-following racing controller with closed-loop speed."""

    def __init__(self):
        super().__init__('gap_follower')

        # --- Perception parameters -------------------------------------------
        self.declare_parameter('fov_deg', 90.0)
        self.declare_parameter('horizon_m', 8.0)
        self.declare_parameter('bubble_radius_m', 0.30)
        self.declare_parameter('clearance_margin_m', 0.055)
        self.declare_parameter('gap_threshold_m', 1.0)
        self.declare_parameter('min_gap_width_m', 0.45)
        self.declare_parameter('disparity_threshold_m', 0.35)
        self.declare_parameter('front_cone_deg', 12.0)

        # --- Gap scoring weights ---------------------------------------------
        self.declare_parameter('w_depth', 1.0)
        self.declare_parameter('w_width', 0.5)
        self.declare_parameter('w_hysteresis', 1.5)

        # --- Speed estimation -------------------------------------------------
        self.declare_parameter('encoder_m_per_unit', WHEEL_RADIUS_M)
        self.declare_parameter('speed_filter_tau_s', 0.15)

        # --- Steering ---------------------------------------------------------
        self.declare_parameter('lookahead_base_m', 0.7)
        self.declare_parameter('lookahead_gain_s', 0.40)
        self.declare_parameter('lookahead_min_m', 1.2)
        self.declare_parameter('lookahead_max_m', 3.5)
        self.declare_parameter('steer_rate_rad_s', 3.0)

        # --- Speed control ----------------------------------------------------
        self.declare_parameter('a_lat_max', 4.0)
        self.declare_parameter('a_decel_max', 2.5)
        self.declare_parameter('v_max', 5.0)
        self.declare_parameter('v_min', 1.0)
        self.declare_parameter('stop_margin_m', 0.4)
        self.declare_parameter('throttle_ff', 0.042)
        self.declare_parameter('throttle_kp', 0.10)
        self.declare_parameter('throttle_max', 0.40)
        self.declare_parameter('throttle_filter_tau_s', 0.10)

        self._reload_parameters()

        # --- State ------------------------------------------------------------
        self.prev_stamp_s = None
        self.prev_steer_rad = 0.0
        self.prev_bearing_rad = 0.0
        self.throttle_filt = 0.0

        self.speed_mps = 0.0
        self.enc_prev_pos = None
        self.enc_prev_stamp_s = None
        self.last_log_s = 0.0

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
        self.left_enc_sub = self.create_subscription(
            JointState, '/autodrive/roboracer_1/left_encoder',
            self.encoder_callback, qos)

        self.get_logger().info('gap_follower v3 ready, waiting for laser scans')

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
        self.front_cone_rad = math.radians(g('front_cone_deg').value)

        self.w_depth = g('w_depth').value
        self.w_width = g('w_width').value
        self.w_hysteresis = g('w_hysteresis').value

        self.encoder_m_per_unit = g('encoder_m_per_unit').value
        self.speed_filter_tau_s = g('speed_filter_tau_s').value

        self.lookahead_base_m = g('lookahead_base_m').value
        self.lookahead_gain_s = g('lookahead_gain_s').value
        self.lookahead_min_m = g('lookahead_min_m').value
        self.lookahead_max_m = g('lookahead_max_m').value
        self.steer_rate_rad_s = g('steer_rate_rad_s').value

        self.a_lat_max = g('a_lat_max').value
        self.a_decel_max = g('a_decel_max').value
        self.v_max = g('v_max').value
        self.v_min = g('v_min').value
        self.stop_margin_m = g('stop_margin_m').value
        self.throttle_ff = g('throttle_ff').value
        self.throttle_kp = g('throttle_kp').value
        self.throttle_max = g('throttle_max').value
        self.throttle_filter_tau_s = g('throttle_filter_tau_s').value

    # -------------------------------------------------------------------------
    def encoder_callback(self, msg: JointState):
        """
        Differentiate wheel encoder position to estimate forward speed.

        The velocity field is not populated by the bridge, so the position
        field is differenced against the previous sample. A first order
        low-pass with a time constant in seconds smooths the result without
        making it dependent on the message rate.
        """
        if not msg.position:
            return

        stamp_s = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        pos = float(msg.position[0])

        if self.enc_prev_stamp_s is None:
            self.enc_prev_stamp_s = stamp_s
            self.enc_prev_pos = pos
            return

        dt = stamp_s - self.enc_prev_stamp_s
        if dt <= 1e-4:
            return

        d_pos = pos - self.enc_prev_pos
        self.enc_prev_stamp_s = stamp_s
        self.enc_prev_pos = pos

        raw = abs(d_pos) * self.encoder_m_per_unit / dt

        # Reject impossible jumps, e.g. a counter reset after a collision.
        if raw > 25.0:
            return

        alpha = dt / max(self.speed_filter_tau_s + dt, 1e-6)
        self.speed_mps += alpha * (raw - self.speed_mps)

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
            self.get_logger().warn('no viable gap found, coasting',
                                   throttle_duration_sec=1.0)
            self._publish(self.prev_steer_rad / MAX_STEER_RAD, 0.0)
            return

        self.prev_bearing_rad = bearing_rad

        # --- Steering ---------------------------------------------------------
        lookahead_m = float(np.clip(
            self.lookahead_base_m + self.lookahead_gain_s * self.speed_mps,
            self.lookahead_min_m,
            self.lookahead_max_m,
        ))
        target_steer_rad = self._pure_pursuit(bearing_rad, lookahead_m)
        steer_rad = self._rate_limit(target_steer_rad, dt)

        # --- Speed ------------------------------------------------------------
        v_target = self._speed_target(target_steer_rad, clearance_m)
        throttle = self._throttle_for(v_target, dt)

        self._log_state(msg, v_target, lookahead_m)
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
        sideways by half the car width plus margin, so a gap the vehicle cannot
        physically fit through is never a candidate.
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

        The score rewards depth and physical width and penalises angular
        distance from the bearing chosen on the previous scan. Without that
        penalty the choice flips between similar candidates on alternate scans,
        which excites a steering limit cycle.
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
        Geometric steering law for an Ackermann vehicle chasing an aim point at
        bearing `bearing_rad` and distance `lookahead_m`:

            delta = atan(2 * L * sin(alpha) / Ld)

        With Ld floored at 1.2 m the argument cannot exceed 0.54, so the result
        stays inside the 0.5236 rad mechanical limit for every bearing and the
        actuator cannot be driven into saturation by geometry alone.
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

    def _speed_target(self, steer_rad, clearance_m):
        """
        Smallest of three limits:

          * Curvature limit. The instantaneous turn radius implied by the
            steering angle is R = L / tan(delta), and holding lateral
            acceleration below a_lat_max gives v = sqrt(a_lat_max * R).
          * Clearance limit. With only idle braking torque available, the
            speed from which the vehicle can still shed all its energy inside
            the visible clearance is v = sqrt(2 * a_decel_max * distance).
          * An absolute cap.
        """
        tan_delta = abs(math.tan(steer_rad))
        if tan_delta < 1e-3:
            v_curve = self.v_max
        else:
            radius = WHEELBASE_M / tan_delta
            v_curve = math.sqrt(self.a_lat_max * radius)

        usable = max(clearance_m - self.stop_margin_m, 0.0)
        v_clear = math.sqrt(2.0 * self.a_decel_max * usable)

        return float(np.clip(min(v_curve, v_clear, self.v_max),
                             0.0, self.v_max))

    def _throttle_for(self, v_target, dt):
        """
        Feedforward plus proportional speed control, low-pass filtered.

        Throttle is a smooth function of speed error rather than a fast
        function of scan geometry, which removes the throttle chatter that the
        v2 steering-angle derate produced. Negative values are never emitted
        because negative throttle engages reverse rather than braking; the
        vehicle decelerates on idle torque alone.
        """
        raw = self.throttle_ff * v_target + self.throttle_kp * (v_target - self.speed_mps)
        raw = float(np.clip(raw, 0.0, self.throttle_max))

        alpha = dt / max(self.throttle_filter_tau_s + dt, 1e-6)
        self.throttle_filt += alpha * (raw - self.throttle_filt)
        return max(self.throttle_filt, 0.0)

    def _log_state(self, msg, v_target, lookahead_m):
        """Emit a one-line state summary about once per second."""
        now_s = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if now_s - self.last_log_s < 1.0:
            return
        self.last_log_s = now_s
        self.get_logger().info(
            'v={:.2f} v_tgt={:.2f} Ld={:.2f} steer={:.3f} thr={:.3f}'.format(
                self.speed_mps, v_target, lookahead_m,
                self.prev_steer_rad, self.throttle_filt))

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
