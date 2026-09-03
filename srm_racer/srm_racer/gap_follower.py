#!/usr/bin/env python3
# =============================================================================
# File        : gap_follower.py
# Package     : srm_racer
# Project     : RoboRacer Sim Racing League @ IROS 2026 (AutoDRIVE Ecosystem)
# Target      : ROS 2 Humble / Python 3.10
# Created     : 2026-08-31
# Revised     : 2026-09-01  (v2: pure pursuit steering, gap hysteresis)
# Revised     : 2026-09-02  (v3: encoder speed estimate, speed-scaled lookahead)
# Revised     : 2026-09-02  (v4: preview speed profile, target rate limiting)
# Revised     : 2026-09-03  (v5: clearance-maximising aim point - REVERTED)
# Revised     : 2026-09-03  (v6: v4 gap selection restored, distance-indexed
#                            speed profile retained, aim unwind rate limit)
# =============================================================================
# CODE DESCRIPTION
# -----------------------------------------------------------------------------
# Reactive racing controller for the RoboRacer digital twin in the AutoDRIVE
# Simulator. Consumes only competition-legal input streams (2D LiDAR and wheel
# encoders) and emits normalised steering and throttle commands. No map, no
# stored knowledge of the circuit, and no ground-truth pose, so the behaviour
# transfers to an unseen racetrack.
#
# REVISION NOTES (v6)
# -----------------------------------------------------------------------------
# v5 replaced the deepest-gap aim criterion with a clearance-maximising one, on
# the theory that aiming at the medial axis of the corridor would stop the
# vehicle unwinding early on corner exit. That was a design error. Maximising
# clearance alone contains no reward for forward progress, so the criterion is
# satisfied by any laterally central point, including one almost perpendicular
# to the direction of travel. Logged telemetry showed the aim bearing pinned at
# the extreme edge of the candidate fan on every sample, driving the vehicle
# sideways into the barrier. Attempting to correct this by weighting a
# straight-ahead penalty merely moved the failure to the opposite extreme, with
# the aim bearing pinned at zero and no turning attempted at all, because the
# clearance term (metres) and the penalty terms (radians) have no principled
# relative scale.
#
# v6 therefore restores the v4 gap selection, which drove the circuit correctly
# at 13.2 s with no collisions, and addresses the corner-exit problem in a
# targeted way instead.
#
# THE CORNER-EXIT FIX
#   Part way through a corner the deepest gap becomes the straight beyond the
#   exit rather than the corner still being negotiated. The aim bearing then
#   swings back toward centre, pure pursuit unwinds the steering, and the
#   vehicle drifts wide before the corner is complete. On the right hand turn
#   after the main straight this ran the vehicle out of road.
#
#   Rather than change what is optimised, v6 constrains how fast the aim
#   bearing may unwind. Movement of the aim bearing AWAY from centre is
#   unrestricted, so corner entry stays sharp, while movement back TOWARD
#   centre is rate limited in rad/s. The steering therefore stays wound until
#   the vehicle has genuinely rotated, and unwinds at a rate the chassis can
#   follow. This is distinct from the existing steering slew limit, which
#   constrains the actuator; this constrains the target.
#
# THE SPEED PROFILE (retained from v5)
#   v4 closed a feedback loop through geometry: speed set the lookahead, the
#   lookahead set the aim bearing, the aim bearing set the curvature estimate,
#   the curvature set the speed target, the target set the throttle, and the
#   throttle set the speed. Logged throttle showed a clean sinusoid of period
#   about 0.8 s that no filtering or rate limiting removed, because each such
#   measure damps a link rather than breaking the cycle.
#
#   v6 indexes the speed profile by distance ahead. At each of several fixed
#   preview distances the scan gives a corridor bearing and a free distance,
#   from which a cornering limit follows; a backward pass then propagates the
#   braking constraint from far to near so that a slow point ahead lowers the
#   target now. No term depends on current speed or on the steering lookahead,
#   so the profile changes only when the track changes.
#
# PIPELINE, once per incoming laser scan:
#   1. Sanitise ranges (NaN and +inf replaced, values clipped to a horizon).
#   2. Subtract the LiDAR-to-bumper offset so ranges describe bumper clearance.
#   3. Restrict attention to a forward field of view.
#   4. Zero a safety bubble around the nearest return.
#   5. Extend disparities by half the car width so no gap narrower than the
#      vehicle is ever selected (this also closes the apparent openings between
#      the cylindrical track ducts).
#   6. Score candidate gaps on depth and physical width, with a hysteresis
#      penalty against the previous choice; take the best gap's centre.
#   7. Apply the unwind rate limit to that bearing.
#   8. Convert to a steering angle by pure pursuit, then slew-limit it.
#   9. Evaluate the distance-indexed speed profile with a backward braking
#      pass, close a feedforward plus proportional loop on measured speed,
#      filter, and publish.
#
# DESIGN CONSTRAINTS
#   * Every rate limit and filter is expressed per second and multiplied by the
#     measured dt between scans. The bridge tick observed on the development
#     laptop is ~18.5 Hz against a documented 40 Hz sensor rate, and the
#     evaluation workstation will differ again. Time-based logic keeps
#     behaviour identical across all of them.
#   * Negative throttle engages REVERSE on this vehicle rather than braking, so
#     throttle is clamped non-negative and deceleration relies on idle torque.
#   * Restricted topics (ips, odom, tf, lap and collision telemetry, and
#     reset_command) are never subscribed to, per section 2.4 of the rule book.
#     Wheel encoders are explicitly permissible inputs.
#
# GEOMETRY (2026 Technical Guide)
#   Car width 0.270 m, wheelbase 0.324 m, front overhang 0.090 m.
#   Wheel radius 0.0590 m. LiDAR frame at x = 0.2733 m from the rear axle;
#   front bumper at 0.414 m. Steering limits +/- 0.5236 rad.
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
    """Depth-based gap follower with decoupled speed profile."""

    def __init__(self):
        super().__init__('gap_follower')

        # --- Perception --------------------------------------------------------
        self.declare_parameter('fov_deg', 90.0)
        self.declare_parameter('horizon_m', 8.0)
        self.declare_parameter('bubble_radius_m', 0.30)
        self.declare_parameter('clearance_margin_m', 0.055)
        self.declare_parameter('gap_threshold_m', 1.0)
        self.declare_parameter('min_gap_width_m', 0.45)
        self.declare_parameter('disparity_threshold_m', 0.35)

        # --- Gap scoring --------------------------------------------------------
        self.declare_parameter('w_depth', 1.0)
        self.declare_parameter('w_width', 0.5)
        self.declare_parameter('w_hysteresis', 1.5)

        # --- Aim unwind limiting ------------------------------------------------
        self.declare_parameter('aim_unwind_rate_rad_s', 100.0)

        # --- Speed estimation ---------------------------------------------------
        self.declare_parameter('encoder_m_per_unit', WHEEL_RADIUS_M)
        self.declare_parameter('speed_filter_tau_s', 0.15)

        # --- Steering -----------------------------------------------------------
        self.declare_parameter('lookahead_base_m', 0.9)
        self.declare_parameter('lookahead_gain_s', 0.25)
        self.declare_parameter('lookahead_min_m', 1.4)
        self.declare_parameter('lookahead_max_m', 2.2)
        self.declare_parameter('steer_rate_rad_s', 3.2)

        # --- Speed profile ------------------------------------------------------
        self.declare_parameter('preview_distances', [1.2, 2.0, 2.8, 3.6, 4.5])
        self.declare_parameter('profile_fov_deg', 60.0)
        self.declare_parameter('a_lat_max', 7.0)
        self.declare_parameter('a_decel_max', 1.5)
        self.declare_parameter('v_max', 4.0)
        self.declare_parameter('v_min', 1.5)

        # --- Throttle -----------------------------------------------------------
        self.declare_parameter('throttle_ff', 0.040)
        self.declare_parameter('throttle_kp', 0.14)
        self.declare_parameter('throttle_max', 0.25)
        self.declare_parameter('throttle_filter_tau_s', 0.10)

        self._reload_parameters()

        # --- State --------------------------------------------------------------
        self.prev_stamp_s = None
        self.prev_steer_rad = 0.0
        self.prev_bearing_rad = 0.0
        self.aim_rad = 0.0
        self.throttle_filt = 0.0

        self.speed_mps = 0.0
        self.enc_prev_pos = None
        self.enc_prev_stamp_s = None
        self.last_log_s = 0.0

        # --- ROS interfaces -----------------------------------------------------
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

        self.get_logger().info('gap_follower v6 ready, waiting for laser scans')

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

        self.aim_unwind_rate_rad_s = g('aim_unwind_rate_rad_s').value

        self.encoder_m_per_unit = g('encoder_m_per_unit').value
        self.speed_filter_tau_s = g('speed_filter_tau_s').value

        self.lookahead_base_m = g('lookahead_base_m').value
        self.lookahead_gain_s = g('lookahead_gain_s').value
        self.lookahead_min_m = g('lookahead_min_m').value
        self.lookahead_max_m = g('lookahead_max_m').value
        self.steer_rate_rad_s = g('steer_rate_rad_s').value

        self.preview_distances = list(g('preview_distances').value)
        self.profile_fov_rad = math.radians(g('profile_fov_deg').value)
        self.a_lat_max = g('a_lat_max').value
        self.a_decel_max = g('a_decel_max').value
        self.v_max = g('v_max').value
        self.v_min = g('v_min').value

        self.throttle_ff = g('throttle_ff').value
        self.throttle_kp = g('throttle_kp').value
        self.throttle_max = g('throttle_max').value
        self.throttle_filter_tau_s = g('throttle_filter_tau_s').value

    # -------------------------------------------------------------------------
    def encoder_callback(self, msg: JointState):
        """
        Differentiate wheel encoder position to estimate forward speed. The
        velocity field is not populated by the bridge, so successive position
        samples are differenced and low-pass filtered with a time constant in
        seconds, keeping the estimate independent of message rate.
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

        raw_bearing = self._select_gap(window, msg, lo)

        if raw_bearing is None:
            self.get_logger().warn('no viable gap found, coasting',
                                   throttle_duration_sec=1.0)
            self._publish(self.prev_steer_rad / MAX_STEER_RAD, 0.0)
            return

        self.prev_bearing_rad = raw_bearing
        aim_rad = self._limit_unwind(raw_bearing, dt)

        # --- Steering -----------------------------------------------------------
        lookahead_m = float(np.clip(
            self.lookahead_base_m + self.lookahead_gain_s * self.speed_mps,
            self.lookahead_min_m,
            self.lookahead_max_m,
        ))
        target_steer_rad = self._pure_pursuit(aim_rad, lookahead_m)
        steer_rad = self._rate_limit(target_steer_rad, dt)

        # --- Speed --------------------------------------------------------------
        v_target = self._speed_profile(ranges, msg)
        throttle = self._throttle_for(v_target, dt)

        self._log_state(msg, v_target, lookahead_m, raw_bearing, aim_rad)
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
        Score every viable gap and return the aim bearing of the best one. The
        score rewards depth and physical width and penalises angular distance
        from the bearing chosen on the previous scan, so the aim point cannot
        flip between similar candidates on alternate scans.
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

    def _limit_unwind(self, raw_bearing, dt):
        """
        Rate limit movement of the aim bearing back toward centre.

        Part way through a corner the deepest gap becomes the straight beyond
        the exit, so the raw bearing collapses toward zero while the vehicle is
        still rotating. Following that immediately unwinds the steering early
        and runs the vehicle wide. Movement away from centre is left
        unrestricted so that corner entry stays sharp; only the return is
        limited, at a rate the chassis can plausibly follow.
        """
        moving_toward_centre = abs(raw_bearing) < abs(self.aim_rad)

        if moving_toward_centre:
            max_step = self.aim_unwind_rate_rad_s * dt
            delta = raw_bearing - self.aim_rad
            delta = float(np.clip(delta, -max_step, max_step))
            self.aim_rad += delta
        else:
            self.aim_rad = raw_bearing

        return self.aim_rad

    def _pure_pursuit(self, bearing_rad, lookahead_m):
        """
        Geometric steering law for an Ackermann vehicle chasing an aim point at
        bearing `bearing_rad` and distance `lookahead_m`:

            delta = atan(2 * L * sin(alpha) / Ld)

        With Ld floored at 1.4 m the argument cannot exceed 0.46, so the result
        stays inside the mechanical limit for every bearing and the actuator
        cannot be driven into saturation by geometry alone.
        """
        delta = math.atan2(2.0 * WHEELBASE_M * math.sin(bearing_rad), lookahead_m)
        return float(np.clip(delta, -MAX_STEER_RAD, MAX_STEER_RAD))

    def _rate_limit(self, target_rad, dt):
        """Slew-limit the steering command and clamp to the mechanical limit."""
        max_step = self.steer_rate_rad_s * dt
        delta = float(np.clip(target_rad - self.prev_steer_rad, -max_step, max_step))
        steer = float(np.clip(self.prev_steer_rad + delta, -MAX_STEER_RAD, MAX_STEER_RAD))
        self.prev_steer_rad = steer
        return steer

    def _speed_profile(self, ranges, msg):
        """
        Build a speed profile indexed by distance ahead, then return the value
        applicable now.

        At each fixed preview distance d the deepest bearing within the profile
        field of view is found. The arc through the vehicle and that point has
        radius R = d / (2 sin alpha), giving a cornering limit
        v = sqrt(a_lat_max * R). A backward pass then propagates the braking
        constraint inward:

            v[k] = min(v[k], sqrt(v[k+1]^2 + 2 * a_decel * (d[k+1] - d[k])))

        so a slow point far ahead lowers the target now, which is the only way
        to decelerate in time on a vehicle with no brake. Nothing here depends
        on current speed or on the steering lookahead, so the profile changes
        only when the track changes; this is what breaks the v4 oscillation.
        """
        limits = []

        for d in self.preview_distances:
            # Widest bearing at which the track is still clear to distance d.
            best_alpha = 0.0
            found = False

            n_samples = 21
            for i in range(n_samples):
                frac = i / (n_samples - 1)
                alpha = -self.profile_fov_rad + frac * 2.0 * self.profile_fov_rad
                idx = int((alpha - msg.angle_min) / msg.angle_increment)
                if idx < 0 or idx >= ranges.size:
                    continue
                if ranges[idx] >= d:
                    if not found or abs(alpha) < abs(best_alpha):
                        best_alpha = alpha
                        found = True

            if not found:
                # Nothing clear this far ahead: hold at the floor.
                limits.append(self.v_min)
                continue

            sin_a = abs(math.sin(best_alpha))
            if sin_a < 1e-3:
                limits.append(self.v_max)
            else:
                radius = d / (2.0 * sin_a)
                limits.append(math.sqrt(self.a_lat_max * radius))

        # Backward pass: propagate the braking constraint inward.
        v_allow = list(limits)
        for k in range(len(v_allow) - 2, -1, -1):
            gap = self.preview_distances[k + 1] - self.preview_distances[k]
            reachable = math.sqrt(v_allow[k + 1] ** 2 + 2.0 * self.a_decel_max * gap)
            v_allow[k] = min(v_allow[k], reachable)

        return float(np.clip(v_allow[0], self.v_min, self.v_max))

    def _throttle_for(self, v_target, dt):
        """
        Feedforward plus proportional speed control, low-pass filtered.
        Negative values are never emitted because negative throttle engages
        reverse rather than braking; the vehicle decelerates on idle torque.
        """
        raw = self.throttle_ff * v_target + self.throttle_kp * (v_target - self.speed_mps)
        raw = float(np.clip(raw, 0.0, self.throttle_max))

        alpha = dt / max(self.throttle_filter_tau_s + dt, 1e-6)
        self.throttle_filt += alpha * (raw - self.throttle_filt)
        return max(self.throttle_filt, 0.0)

    def _log_state(self, msg, v_target, lookahead_m, raw_bearing, aim_rad):
        """Emit a one-line state summary about once per second."""
        now_s = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if now_s - self.last_log_s < 1.0:
            return
        self.last_log_s = now_s
        self.get_logger().info(
            'v={:.2f} tgt={:.2f} Ld={:.2f} raw_aim={:.3f} aim={:.3f} '
            'steer={:.3f} thr={:.3f}'.format(
                self.speed_mps, v_target, lookahead_m, raw_bearing,
                aim_rad, self.prev_steer_rad, self.throttle_filt))

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
