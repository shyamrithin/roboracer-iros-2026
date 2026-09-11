#!/usr/bin/env python3
# =============================================================================
# File        : dead_reckoning.py
# Package     : srm_racer
# Project     : RoboRacer Sim Racing League @ IROS 2026 (AutoDRIVE Ecosystem)
# Target      : ROS 2 Humble / Python 3.10
# Created     : 2026-09-06
# =============================================================================
# CODE DESCRIPTION
# -----------------------------------------------------------------------------
# Dead-reckoned pose estimation from wheel encoders and IMU, together with a
# live comparison against simulator ground truth for validation purposes.
#
# PURPOSE
# -----------------------------------------------------------------------------
# The present racing stack is purely reactive: it sees eight metres of LiDAR and
# responds to it, with no notion of where it is on the circuit. That bounds the
# achievable lap time, because a racing line requires knowing where a corner
# ends before entering it, and published benchmarks report that offline
# trajectory optimisation and tracking is substantially faster than
# follow-the-gap for exactly this reason.
#
# A raceline approach needs a pose estimate. This node is the first component of
# that, and it is deliberately built and measured in isolation before anything
# is allowed to depend on it.
#
# METHOD
# -----------------------------------------------------------------------------
# Heading is taken directly from the IMU, which supplies 3-DOF orientation as
# Euler angles; the yaw component is used as-is rather than integrated from
# angular rate, which avoids accumulating gyro bias.
#
# Distance travelled is obtained by differencing the wheel encoder position,
# which reports wheel angle in radians, so the increment is r * dtheta with
# r = 0.0590 m. This was validated in an earlier revision: the resulting speed
# estimate tracked the simulator's own speed readout closely.
#
# Position is then the running sum
#     x += d * cos(yaw)
#     y += d * sin(yaw)
# evaluated per encoder sample.
#
# The estimate will drift. Encoder distance is corrupted by wheel slip, which
# this vehicle experiences readily given a lateral tire curve that peaks at 0.01
# slip, and any heading error rotates all subsequent displacement. The purpose
# of this node is to quantify that drift rather than to assume it is tolerable:
# the rate of growth determines whether scan matching against a prior map has
# enough of an initial guess to converge.
#
# LEGALITY
# -----------------------------------------------------------------------------
# Wheel encoders and the IMU are permissible inputs at run time. The IPS topic
# is restricted, and is subscribed to here solely for validation, which section
# 2.4 of the rule book explicitly permits for debugging. No component of the
# racing stack reads it, and this node is not launched during a race.
# =============================================================================

import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Point

WHEEL_RADIUS_M = 0.0590


class DeadReckoning(Node):
    """Encoder and IMU dead reckoning with ground-truth error reporting."""

    def __init__(self):
        super().__init__('dead_reckoning')

        self.declare_parameter('encoder_m_per_unit', WHEEL_RADIUS_M)
        self.encoder_m_per_unit = self.get_parameter('encoder_m_per_unit').value

        # Estimated pose.
        self.x = None
        self.y = None
        self.yaw = 0.0
        self.have_yaw = False

        self.enc_prev = None
        self.distance_travelled = 0.0

        # Ground truth, for validation only.
        self.true_x = None
        self.true_y = None

        self.last_log_s = 0.0

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.create_subscription(
            Imu, '/autodrive/roboracer_1/imu', self.imu_callback, qos)
        self.create_subscription(
            JointState, '/autodrive/roboracer_1/left_encoder',
            self.encoder_callback, qos)
        self.create_subscription(
            Point, '/autodrive/roboracer_1/ips', self.ips_callback, qos)

        self.get_logger().info('dead_reckoning ready')

    def imu_callback(self, msg: Imu):
        """Take yaw directly from the reported orientation quaternion."""
        q = msg.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.yaw = math.atan2(siny, cosy)
        self.have_yaw = True

    def ips_callback(self, msg: Point):
        """Record ground truth and seed the estimate on the first sample."""
        self.true_x = msg.x
        self.true_y = msg.y
        if self.x is None:
            self.x = msg.x
            self.y = msg.y

    def encoder_callback(self, msg: JointState):
        """Integrate wheel distance along the current heading."""
        if not msg.position or not self.have_yaw or self.x is None:
            return

        pos = float(msg.position[0])
        if self.enc_prev is None:
            self.enc_prev = pos
            return

        d = (pos - self.enc_prev) * self.encoder_m_per_unit
        self.enc_prev = pos

        # Reject implausible jumps, e.g. a counter reset after a collision.
        if abs(d) > 1.0:
            return

        self.x += d * math.cos(self.yaw)
        self.y += d * math.sin(self.yaw)
        self.distance_travelled += abs(d)

        self._log(msg)

    def _log(self, msg):
        """Report estimated pose, true pose, and error once per second."""
        stamp_s = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if stamp_s - self.last_log_s < 1.0:
            return
        self.last_log_s = stamp_s

        if self.true_x is None:
            return

        err = math.hypot(self.x - self.true_x, self.y - self.true_y)
        pct = 100.0 * err / max(self.distance_travelled, 1e-3)

        self.get_logger().info(
            'est=({:+.2f},{:+.2f})  true=({:+.2f},{:+.2f})  '
            'err={:.3f} m  travelled={:.1f} m  drift={:.2f}%'.format(
                self.x, self.y, self.true_x, self.true_y,
                err, self.distance_travelled, pct))


def main(args=None):
    rclpy.init(args=args)
    node = DeadReckoning()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
