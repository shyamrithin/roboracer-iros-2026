#!/usr/bin/env python3
# =============================================================================
# File        : dead_reckoning.py
# Package     : srm_racer
# Project     : RoboRacer Sim Racing League @ IROS 2026 (AutoDRIVE Ecosystem)
# Target      : ROS 2 Humble / Python 3.10
# Revised     : 2026-09-11  (v2: both encoders averaged, odometry published,
#                            initial pose from parameters rather than IPS)
# =============================================================================
# CODE DESCRIPTION
# -----------------------------------------------------------------------------
# Dead-reckoned pose estimation from wheel encoders and IMU yaw, published as
# nav_msgs/Odometry so it can stand in for the simulator's ground-truth
# odometry at run time.
#
# Only permissible run-time inputs are used: wheel encoders and IMU. The IPS
# topic is subscribed to for validation only, and only when explicitly
# enabled, since it is restricted during a race.
#
# WHY v2 EXISTS
# -----------------------------------------------------------------------------
# v1 integrated distance from the LEFT ENCODER ALONE. A single wheel does not
# travel the same arc as the vehicle centre: on a turn of radius R the inner
# wheel covers (R - t/2)/R of the centre distance and the outer wheel
# (R + t/2)/R. With a 0.236 m rear track and a 0.561 m minimum turn radius
# that is a 21 per cent error at full lock, signed by turn direction.
#
# Porto is not symmetric, so those errors do not cancel over a lap. They
# accumulate with consistent sign and vary with track position - which is
# exactly the reported v1 behaviour: bounded 1.5 to 2.2 m error in y, same
# sign throughout, varying around the circuit. That was previously read as a
# fixed rotational offset acquired at initialisation, but heading_fit.py has
# since shown yaw needs no offset at all (parts[10], sign +1, 0.57 deg
# residual), so the rotational explanation does not hold.
#
# THE CORRECTION
# -----------------------------------------------------------------------------
# The centre of the rear axle travels the mean of the two wheel distances:
#
#     d = 0.5 * (d_left + d_right)
#
# This is exact for a rigid axle regardless of turn radius, and it does NOT
# involve the track width - track width only enters if yaw rate is derived
# from the wheel difference, and here yaw comes from the IMU instead. So the
# discrepancy between the documented 0.250 m track and the devkit's 0.236 m
# does not affect this estimate.
#
# ASYNCHRONOUS ENCODERS
# -----------------------------------------------------------------------------
# The two encoders publish on separate topics and their callbacks do not
# arrive together. Rather than trying to pair samples, each wheel's cumulative
# distance is tracked independently and the centre distance is recomputed on
# every callback from whichever wheel fired. The increment integrated is the
# change in that centre distance, so no sample is double counted and no
# pairing is needed.
#
# INITIAL POSE
# -----------------------------------------------------------------------------
# v1 seeded x and y from the IPS topic, which is restricted at run time. v2
# takes the initial pose from parameters, which is how it must work in a race.
# Set them to the same values used for map_start_pose in the localization
# config so the two agree.
#
# COLLISION RESETS
# -----------------------------------------------------------------------------
# The rule book states that each collision resets the vehicle to the last
# checkpoint. Encoders keep counting across that teleport, so this estimate
# will be wrong afterwards by the reset distance. Recovery has to come from
# the scan matcher, which is one reason the localiser must not lean on this
# estimate too heavily.
# =============================================================================

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Imu, JointState
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry

# --- Fixed vehicle geometry, do not tune -------------------------------------
WHEEL_RADIUS_M = 0.0590


class DeadReckoning(Node):
    """Pose estimate from averaged wheel encoders and IMU yaw."""

    def __init__(self):
        super().__init__('dead_reckoning')

        self.declare_parameter('encoder_m_per_unit', WHEEL_RADIUS_M)
        self.declare_parameter('initial_x', 0.0)
        self.declare_parameter('initial_y', 0.0)
        self.declare_parameter('use_ips_validation', False)
        self.declare_parameter('odom_topic', '/dead_reckoning/odom')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('max_step_m', 1.0)

        g = self.get_parameter
        self.encoder_m_per_unit = g('encoder_m_per_unit').value
        self.use_ips_validation = g('use_ips_validation').value
        self.odom_frame = g('odom_frame').value
        self.base_frame = g('base_frame').value
        self.max_step_m = g('max_step_m').value

        # --- State ------------------------------------------------------------
        self.x = g('initial_x').value
        self.y = g('initial_y').value
        self.yaw = 0.0
        self.have_yaw = False

        self.enc_ref = {'left': None, 'right': None}
        self.enc_dist = {'left': 0.0, 'right': 0.0}
        self.centre_prev = 0.0
        self.distance_travelled = 0.0

        self.true_x = None
        self.true_y = None
        self.last_log_s = 0.0

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.odom_pub = self.create_publisher(
            Odometry, g('odom_topic').value, 10)

        self.create_subscription(
            Imu, '/autodrive/roboracer_1/imu', self.imu_callback, qos)
        self.create_subscription(
            JointState, '/autodrive/roboracer_1/left_encoder',
            lambda m: self.encoder_callback(m, 'left'), qos)
        self.create_subscription(
            JointState, '/autodrive/roboracer_1/right_encoder',
            lambda m: self.encoder_callback(m, 'right'), qos)

        if self.use_ips_validation:
            self.create_subscription(
                Point, '/autodrive/roboracer_1/ips', self.ips_callback, qos)
            self.get_logger().warn(
                'IPS validation enabled. This topic is RESTRICTED at run '
                'time and must be disabled before submission.')

        self.get_logger().info(
            'dead_reckoning v2 ready. Both encoders averaged. '
            f'Initial pose ({self.x:.4f}, {self.y:.4f}), yaw from IMU.')

    def imu_callback(self, msg: Imu):
        """Take yaw directly from the reported orientation quaternion."""
        q = msg.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.yaw = math.atan2(siny, cosy)
        self.have_yaw = True

    def ips_callback(self, msg: Point):
        """Record ground truth for validation only. Never used in the estimate."""
        self.true_x = msg.x
        self.true_y = msg.y

    def encoder_callback(self, msg: JointState, side: str):
        """
        Update one wheel's cumulative distance, then integrate the change in
        the averaged centre distance along the current heading.
        """
        if not msg.position or not self.have_yaw:
            return

        pos = float(msg.position[0])

        if self.enc_ref[side] is None:
            self.enc_ref[side] = pos
            return

        self.enc_dist[side] = (pos - self.enc_ref[side]) * self.encoder_m_per_unit

        # Wait until both wheels have a reference before integrating, so the
        # first increment is not half of a single wheel's travel.
        if self.enc_ref['left'] is None or self.enc_ref['right'] is None:
            return

        centre = 0.5 * (self.enc_dist['left'] + self.enc_dist['right'])
        d = centre - self.centre_prev
        self.centre_prev = centre

        # Reject implausible jumps, e.g. a counter reset after a collision.
        if abs(d) > self.max_step_m:
            return

        self.x += d * math.cos(self.yaw)
        self.y += d * math.sin(self.yaw)
        self.distance_travelled += abs(d)

        self.publish_odom(msg.header.stamp)
        self._log(msg)

    def publish_odom(self, stamp):
        """Emit the estimate as nav_msgs/Odometry for the localisation stack."""
        msg = Odometry()
        msg.header.stamp = stamp
        msg.header.frame_id = self.odom_frame
        msg.child_frame_id = self.base_frame
        msg.pose.pose.position.x = self.x
        msg.pose.pose.position.y = self.y
        msg.pose.pose.position.z = 0.0
        msg.pose.pose.orientation.z = math.sin(self.yaw * 0.5)
        msg.pose.pose.orientation.w = math.cos(self.yaw * 0.5)
        self.odom_pub.publish(msg)

    def _log(self, msg):
        """Report the estimate, and the error against truth when validating."""
        now_s = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if now_s - self.last_log_s < 1.0:
            return
        self.last_log_s = now_s

        line = ('est=({:+.3f},{:+.3f}) yaw={:+.3f} dist={:.1f} '
                'L={:+.2f} R={:+.2f}').format(
            self.x, self.y, self.yaw, self.distance_travelled,
            self.enc_dist['left'], self.enc_dist['right'])

        if self.true_x is not None:
            err = math.hypot(self.x - self.true_x, self.y - self.true_y)
            line += ' true=({:+.3f},{:+.3f}) err={:.3f}'.format(
                self.true_x, self.true_y, err)

        self.get_logger().info(line)


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
