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
        self.declare_parameter('path_half_width_m', 0.32)
        self.declare_parameter('arc_max_m', 8.0)

        self._reload_parameters()
        self.last_log_s = 0.0

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
        self.path_half_width_m = g('path_half_width_m').value
        self.arc_max_m = g('arc_max_m').value

    def scan_callback(self, msg: LaserScan):
        """Main control loop, executed once per laser scan."""
        self._reload_parameters()

        ranges, angles = self._prepare(msg)
        if ranges.size == 0:
            self._publish(0.0, 0.0)
            return

        self._extend_disparities(ranges, msg.angle_increment)

        best = self._select_heading(ranges, angles)
        target_rad = float(angles[best])
        steer_norm = self._steering_command(target_rad)

        depth_m = self._path_depth(ranges, angles, steer_norm)
        throttle = self._throttle_for(depth_m, steer_norm)

        self._log_state(msg, target_rad, depth_m, steer_norm, throttle)
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

    def _path_depth(self, ranges, angles, steer_norm):
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

        # Cut power in proportion to steering angle. The lateral tire curve
        # peaks at 0.01 slip and has halved by 0.10, so this vehicle has a very
        # narrow grip window; applying drive torque while cornering hard pushes
        # it past the peak and the car runs wide on exit.
        throttle *= (1.0 - self.steer_derate * abs(steer_norm))
        return max(throttle, 0.0)

    def _log_state(self, msg, target_rad, depth_m, steer_norm, throttle):
        """Emit a one-line state summary about once per second."""
        now_s = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if now_s - self.last_log_s < 1.0:
            return
        self.last_log_s = now_s
        self.get_logger().info(
            'aim={:+.3f} rad  depth={:.2f} m  steer={:+.3f}  thr={:.3f}'.format(
                target_rad, depth_m, steer_norm, throttle))

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
