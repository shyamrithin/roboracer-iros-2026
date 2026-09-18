#!/usr/bin/env python3
# =============================================================================
# ips_tf_shim.py
# RoboRacer IROS 2026 / CEM Navigators
# =============================================================================
#
# CODE DESCRIPTION
# -----------------------------------------------------------------------------
# Broadcasts the ground-truth vehicle pose as map -> odom -> base_link, so that
# raceline_tracker.py can be tested with PERFECT localisation.
#
# This exists to answer one question before any effort goes into building a
# localiser: is the stored raceline actually faster than the reactive stack?
# If the tracker cannot hold the line even with exact position knowledge, no
# amount of particle filtering will help and the whole direction is dead.
# If it can, the predicted 9.25 s against 13.5 s measured is worth chasing.
#
# OFFLINE VALIDATION TOOL - NEVER PART OF THE SUBMITTED CONTAINER
# ---------------------------------------------------------------------------
# It reads /autodrive/roboracer_1/ips, which is simulation ground truth.
# Rules section 2.4 prohibits utilizing ground truth data; the organisers
# confirmed (Slack, 2026-09-17) only that OFFLINE map and raceline preparation
# is permitted. Driving on ground truth is NOT permitted and any lap time
# measured this way is a diagnostic, not a result. Keep this in tools/.
#
# FRAME CONVENTION
#   The map built by scans_to_map.py is in the IPS world frame directly - the
#   scans were projected using these very poses - so no transform is needed
#   between them. map -> odom is published as identity and odom -> base_link
#   carries the pose. This keeps the same two-step chain slam_toolbox would
#   produce, so the tracker's TF lookup is unchanged.
#
# USAGE
#   export ROS_DOMAIN_ID=0
#   export FASTRTPS_DEFAULT_PROFILES_FILE=$HOME/roboracer/devkit_src/tools/fastdds_udp.xml
#   python3 ips_tf_shim.py
#
#   Then, in another terminal, the tracker:
#     python3 raceline_tracker.py ~/mapdata/raceline_bridge.csv
#
#   The gap follower must NOT be running - both publish throttle and steering.
#
# DEPENDENCIES: rclpy, tf2_ros
# =============================================================================

import math
import sys

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Point, TransformStamped
from sensor_msgs.msg import Imu
from tf2_ros import TransformBroadcaster


class IpsTfShim(Node):
    def __init__(self):
        super().__init__('ips_tf_shim')
        self.br = TransformBroadcaster(self)

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=1)

        self.x = self.y = 0.0
        self.qz = 0.0
        self.qw = 1.0
        self.have_ips = False
        self.have_imu = False
        self.n = 0

        self.create_subscription(
            Point, '/autodrive/roboracer_1/ips', self._on_ips, qos)
        self.create_subscription(
            Imu, '/autodrive/roboracer_1/imu', self._on_imu, qos)
        self.create_timer(0.01, self._broadcast)      # 100 Hz
        self.create_timer(2.0, self._report)

        self.get_logger().warn(
            'GROUND TRUTH TF - offline validation only, never for a race run')

    def _on_ips(self, msg):
        self.x, self.y = msg.x, msg.y
        self.have_ips = True

    def _on_imu(self, msg):
        # keep only the yaw component, the vehicle is planar
        q = msg.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.qz = math.sin(yaw * 0.5)
        self.qw = math.cos(yaw * 0.5)
        self.have_imu = True

    def _broadcast(self):
        if not (self.have_ips and self.have_imu):
            return
        now = self.get_clock().now().to_msg()

        # map -> odom, identity: the map was built in the IPS world frame
        t1 = TransformStamped()
        t1.header.stamp = now
        t1.header.frame_id = 'map'
        t1.child_frame_id = 'odom'
        t1.transform.rotation.w = 1.0

        # odom -> base_link, the pose itself
        t2 = TransformStamped()
        t2.header.stamp = now
        t2.header.frame_id = 'odom'
        t2.child_frame_id = 'base_link'
        t2.transform.translation.x = self.x
        t2.transform.translation.y = self.y
        t2.transform.rotation.z = self.qz
        t2.transform.rotation.w = self.qw

        self.br.sendTransform([t1, t2])
        self.n += 1

    def _report(self):
        if not (self.have_ips and self.have_imu):
            self.get_logger().warn(
                f'waiting: ips={self.have_ips} imu={self.have_imu}')
            return
        yaw = 2.0 * math.atan2(self.qz, self.qw)
        self.get_logger().info(
            f'tf {self.n}  pose ({self.x:+.2f}, {self.y:+.2f}) '
            f'yaw {yaw:+.3f}')


def main():
    rclpy.init()
    node = IpsTfShim()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
