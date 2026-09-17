#!/usr/bin/env python3
# =============================================================================
# raceline_tracker.py
# -----------------------------------------------------------------------------
# Project : RoboRacer Sim Racing League @ IROS 2026 - Team CEM Navigators (26)
# Author  : Shyam Rithin
# Location: ~/roboracer/tools/raceline_tracker.py
# Usage   : python3 raceline_tracker.py ~/mapdata/raceline.csv
# =============================================================================
#
# DESCRIPTION
# -----------
# Follows the stored raceline using pure pursuit, taking the vehicle pose from
# the map -> base_link transform that slam_toolbox publishes, and the target
# speed from the waypoint file.
#
# Runs on the HOST alongside slam_bridge and slam_toolbox, with the same
# ROS_DOMAIN_ID and FASTRTPS_DEFAULT_PROFILES_FILE exports. It publishes
# steering and throttle, so the gap follower must NOT be running.
#
# HOW IT DIFFERS FROM THE REACTIVE STACK
# --------------------------------------
# The steering law is the same pure pursuit the gap follower uses. What
# changes is where the aim point comes from and how throttle is set.
#
#   Reactive : aim at the deepest heading in the current scan; throttle
#              constant, cut by steering angle. Speed is near-constant because
#              the vehicle cannot see far enough ahead to plan.
#   Tracking : aim at a point a lookahead distance along a stored line;
#              throttle closes a loop on the speed that line asks for. Speed
#              varies because the curvature of the whole lap is known in
#              advance.
#
# The lap time is expected to come almost entirely from the second difference.
#
# THROTTLE
# --------
# Feedforward plus proportional on measured speed:
#
#     throttle = ff * v_target + kp * (v_target - v_measured)
#
# ff defaults to 0.053, from the measured operating point of the reactive
# stack: 0.22 throttle produced 4.15 m/s over a 31.36 m lap in 7.55 s. That is
# one point on a curve that is certainly not linear, so the proportional term
# has real work to do at the extremes of the profile.
#
# Negative throttle is never emitted: it engages reverse on this vehicle
# rather than braking. Deceleration is by drivetrain drag, measured at
# 4.3 m/s^2 mean over 25 coastdown cuts, and the velocity profile was
# generated with 4.1 to leave margin.
#
# WHAT THIS VERSION DOES NOT DO
# -----------------------------
# There is no fallback to reactive control. If localisation loses the track,
# a path tracker will steer confidently at a line that is not where it thinks
# it is. The only protection here is a stop: if the nearest waypoint is
# further than `lost_distance_m`, throttle goes to zero and steering holds.
# That prevents driving into a wall while the failure is diagnosed; it does
# not recover the lap. Add the reactive fallback once the failure mode is
# understood.
# =============================================================================

import math
import os
import sys

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import JointState
from std_msgs.msg import Float32
from tf2_ros import Buffer, TransformListener

# --- Fixed vehicle geometry, do not tune -------------------------------------
WHEELBASE_M = 0.324
MAX_STEER_RAD = 0.5236
WHEEL_RADIUS_M = 0.0590


def load_raceline(path):
    xs, ys, vs = [], [], []
    with open(path) as fh:
        fh.readline()                      # header
        for line in fh:
            line = line.strip()
            if not line:
                continue
            p = line.split(",")
            xs.append(float(p[0]))
            ys.append(float(p[1]))
            vs.append(float(p[2]))
    return np.array(xs), np.array(ys), np.array(vs)


class RacelineTracker(Node):

    def __init__(self, wx, wy, wv):
        super().__init__("raceline_tracker")

        self.declare_parameter("lookahead_base_m", 0.8)
        self.declare_parameter("lookahead_gain_s", 0.25)
        self.declare_parameter("lookahead_min_m", 1.0)
        self.declare_parameter("lookahead_max_m", 3.0)
        self.declare_parameter("throttle_ff", 0.053)
        self.declare_parameter("throttle_kp", 0.08)
        self.declare_parameter("throttle_max", 0.60)
        self.declare_parameter("speed_scale", 1.0)
        self.declare_parameter("lost_distance_m", 1.5)
        self.declare_parameter("throttle_rate", 0.8)
        self.declare_parameter("control_hz", 50.0)
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_link")

        self.wx, self.wy, self.wv = wx, wy, wv
        self.n = len(wx)
        self.ds = 31.36 / self.n
        self.idx = None
        self.throttle_prev = 0.0

        self.enc_ref = {"left": None, "right": None}
        self.enc_dist = {"left": 0.0, "right": 0.0}
        self.centre_prev = 0.0
        self.prev_t = None
        self.speed = 0.0

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.steer_pub = self.create_publisher(
            Float32, "/autodrive/roboracer_1/steering_command", 10)
        self.throttle_pub = self.create_publisher(
            Float32, "/autodrive/roboracer_1/throttle_command", 10)

        self.create_subscription(
            JointState, "/autodrive/roboracer_1/left_encoder",
            lambda m: self.enc_cb(m, "left"), qos)
        self.create_subscription(
            JointState, "/autodrive/roboracer_1/right_encoder",
            lambda m: self.enc_cb(m, "right"), qos)

        hz = self.get_parameter("control_hz").value
        self.create_timer(1.0 / hz, self.control)
        self.last_log = 0.0
        self.warned = False

        self.get_logger().info(
            f"raceline_tracker ready. {self.n} waypoints, "
            f"target speed {wv.min():.2f} to {wv.max():.2f} m/s. "
            "The gap follower must not be running.")

    # ---- measured speed, averaged over both wheels -------------------------
    def enc_cb(self, msg, side):
        if not msg.position:
            return
        pos = float(msg.position[0])
        if self.enc_ref[side] is None:
            self.enc_ref[side] = pos
            return
        self.enc_dist[side] = (pos - self.enc_ref[side]) * WHEEL_RADIUS_M
        if self.enc_ref["left"] is None or self.enc_ref["right"] is None:
            return
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        centre = 0.5 * (self.enc_dist["left"] + self.enc_dist["right"])
        # Both encoders publish from the same bridge cycle, microseconds
        # apart. Computing speed per callback loses one wheel's increment to
        # the dt guard, halving the estimate. Accumulate over a fixed window
        # instead, so every increment is counted exactly once.
        if self.prev_t is None:
            self.prev_t = t
            self.centre_prev = centre
            return
        dt = t - self.prev_t
        if dt >= 0.08:
            raw = (centre - self.centre_prev) / dt
            if abs(raw) < 25.0:
                self.speed += 0.5 * (raw - self.speed)
            self.centre_prev = centre
            self.prev_t = t

    # ---- nearest waypoint, searched locally once acquired -------------------
    def nearest(self, x, y):
        if self.idx is None:
            d = np.hypot(self.wx - x, self.wy - y)
            return int(np.argmin(d)), float(d.min())
        # The vehicle cannot have moved far since the last cycle, so search a
        # window forward of the previous index. This also stops the tracker
        # latching onto a nearby part of the line it has not reached yet,
        # which on a circuit that doubles back is a real hazard.
        w = 8
        cand = [(self.idx + k) % self.n for k in range(-5, w)]
        d = np.hypot(self.wx[cand] - x, self.wy[cand] - y)
        j = int(np.argmin(d))
        return cand[j], float(d[j])

    def control(self):
        p = self.get_parameter
        try:
            tf = self.tf_buffer.lookup_transform(
                p("map_frame").value, p("base_frame").value,
                rclpy.time.Time())
        except Exception:
            if not self.warned:
                self.get_logger().warn(
                    "waiting for map -> base_link. Is slam_toolbox running "
                    "and localised?")
                self.warned = True
            return
        self.warned = False

        x = tf.transform.translation.x
        y = tf.transform.translation.y
        q = tf.transform.rotation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))

        i, dist = self.nearest(x, y)
        self.idx = i

        if dist > p("lost_distance_m").value:
            self._publish(0.0, 0.0)
            self.get_logger().warn(
                f"nearest waypoint {dist:.2f} m away, stopping. "
                "Localisation has probably lost the track.",
                throttle_duration_sec=1.0)
            return

        v_meas = self.speed
        # Schedule the lookahead on the PLANNED speed, not the measured one.
        # Measured speed oscillates when the throttle cycles, and with no
        # brake that cycling is violent: zero throttle is a 4.3 m/s^2 retard,
        # so the loop swings between extremes. Scheduling on v_target keeps
        # the steering geometry steady while the speed loop settles.
        ld = float(np.clip(
            p("lookahead_base_m").value + p("lookahead_gain_s").value * v_meas,
            p("lookahead_min_m").value, p("lookahead_max_m").value))

        # Walk forward along the line until the lookahead distance is exceeded.
        j = i
        for _ in range(self.n):
            j = (j + 1) % self.n
            if math.hypot(self.wx[j] - x, self.wy[j] - y) >= ld:
                break

        # Aim point in the vehicle frame.
        dx = self.wx[j] - x
        dy = self.wy[j] - y
        c, s = math.cos(-yaw), math.sin(-yaw)
        ax = c * dx - s * dy
        ay = s * dx + c * dy
        alpha = math.atan2(ay, ax)

        delta = math.atan2(2.0 * WHEELBASE_M * math.sin(alpha), ld)
        steer = float(np.clip(delta / MAX_STEER_RAD, -1.0, 1.0))

        v_target = float(self.wv[i]) * p("speed_scale").value
        raw = (p("throttle_ff").value * v_target
               + p("throttle_kp").value * (v_target - v_meas))
        throttle = float(np.clip(raw, 0.0, p("throttle_max").value))



        self._publish(steer, throttle)

        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self.last_log >= 1.0:
            self.last_log = now
            self.get_logger().info(
                "wp={:3d} xte={:.2f} Ld={:.2f} v_t={:.2f} v_m={:.2f} "
                "thr={:.3f} steer={:+.3f}".format(
                    i, dist, ld, v_target, v_meas, throttle, steer))

    def _publish(self, steer, throttle):
        m = Float32()
        m.data = float(np.clip(steer, -1.0, 1.0))
        self.steer_pub.publish(m)
        m2 = Float32()
        m2.data = float(np.clip(throttle, 0.0, 1.0))
        self.throttle_pub.publish(m2)


def main():
    if len(sys.argv) < 2:
        print("usage: python3 raceline_tracker.py <raceline.csv>")
        sys.exit(1)
    path = os.path.expanduser(sys.argv[1])
    if not os.path.isfile(path):
        print(f"error: no such file: {path}")
        sys.exit(1)

    wx, wy, wv = load_raceline(path)
    print(f"loaded {len(wx)} waypoints, "
          f"speed {wv.min():.2f} to {wv.max():.2f} m/s")

    rclpy.init()
    node = RacelineTracker(wx, wy, wv)
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


if __name__ == "__main__":
    main()
