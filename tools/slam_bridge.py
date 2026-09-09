#!/usr/bin/env python3
# =============================================================================
# slam_bridge.py
# -----------------------------------------------------------------------------
# Project : RoboRacer Sim Racing League @ IROS 2026 - Team CEM Navigators (26)
# Author  : Shyam Rithin
# Location: ~/roboracer/tools/slam_bridge.py
# Usage   : python3 ~/roboracer/tools/slam_bridge.py
# =============================================================================
#
# DESCRIPTION
# -----------
# Adapts the AutoDRIVE RoboRacer devkit's topics and TF tree into the shape
# slam_toolbox expects. Runs on the HOST laptop, not inside the competition
# container -- the container is never modified.
#
# The devkit publishes:
#     world -> roboracer_1 -> {lidar, ips, imu, wheels...}
#     /autodrive/roboracer_1/lidar   (LaserScan, frame_id "lidar")
#     /autodrive/roboracer_1/odom    (Odometry, ground-truth pose)
#
# slam_toolbox expects:
#     map -> odom -> base_link -> <scan frame>
#     /scan
#
# This node fills the gap:
#     * republishes the LaserScan on /scan with frame_id "laser"
#     * publishes odom -> base_link from the devkit's odometry
#     * publishes a static base_link -> laser at the documented mount point
#
# WHY NEW FRAME NAMES
# -------------------
# A TF frame may have exactly one parent. "lidar" is already parented to
# "roboracer_1" by the devkit's own broadcaster, so re-parenting it here would
# produce a conflicting tree and break the devkit's transforms. Publishing a
# parallel "laser" frame leaves the devkit's tree untouched and gives
# slam_toolbox a clean chain of its own.
#
# GEOMETRY (from the devkit's broadcast_transforms, verified against the
# running container)
#     lidar mount : [0.2733, 0.0, 0.096] from the vehicle frame,
#                   identity rotation -- no mount rotation to correct
#     vehicle frame: centre of the rear axle
#
# ON USING GROUND-TRUTH ODOMETRY
# ------------------------------
# The devkit's /odom carries the simulator's true pose. Using it here is
# legal: the organisers confirmed a map may be built entirely before the race
# and shipped in the container; only run-time inference is restricted. It also
# gives slam_toolbox a near-perfect motion prior, so any residual error in the
# resulting map is attributable to the scans rather than to odometry.
#
# NOTE ON ROS_DOMAIN_ID
# ---------------------
# The container runs on the default domain (0). If your shell exports
# ROS_DOMAIN_ID=42 (as your PX4/Gazebo setup does), this node will see no
# topics at all. Export ROS_DOMAIN_ID=0 before running. Tailscale must also be
# down, as it disrupts DDS discovery.
# =============================================================================

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, ReliabilityPolicy, HistoryPolicy,
                       DurabilityPolicy)

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster

# ---- Frame names -----------------------------------------------------------
ODOM_FRAME = "odom"
BASE_FRAME = "base_link"
LASER_FRAME = "laser"

# ---- Devkit topics ---------------------------------------------------------
TOPIC_SCAN_IN = "/autodrive/roboracer_1/lidar"
TOPIC_ODOM_IN = "/autodrive/roboracer_1/odom"
TOPIC_SCAN_OUT = "/scan"

# ---- LiDAR mount, from the devkit TF tree ----------------------------------
LIDAR_X = 0.2733
LIDAR_Y = 0.0
LIDAR_Z = 0.096


class SlamBridge(Node):

    def __init__(self):
        super().__init__("slam_bridge")

        # Matched exactly to the devkit's publisher QoS (RELIABLE,
        # VOLATILE, KEEP_LAST, depth 1). A BEST_EFFORT subscriber against a
        # RELIABLE publisher is nominally compatible, but matching removes
        # the variable entirely.
        qos_in = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        # /scan published RELIABLE: that serves both RELIABLE and
        # BEST_EFFORT subscribers, so it works whichever slam_toolbox asks
        # for.
        qos_out = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_broadcaster = StaticTransformBroadcaster(self)
        self.publish_static_transform()

        self.scan_pub = self.create_publisher(LaserScan, TOPIC_SCAN_OUT,
                                              qos_out)
        self.create_subscription(LaserScan, TOPIC_SCAN_IN,
                                 self.on_scan, qos_in)
        self.create_subscription(Odometry, TOPIC_ODOM_IN,
                                 self.on_odom, qos_in)

        self.n_scan = 0
        self.n_odom = 0
        self.create_timer(5.0, self.report)

        self.get_logger().info(
            f"slam_bridge up. {TOPIC_SCAN_IN} -> {TOPIC_SCAN_OUT} "
            f"(frame '{LASER_FRAME}'), {ODOM_FRAME} -> {BASE_FRAME} "
            f"-> {LASER_FRAME}")

    def publish_static_transform(self):
        """base_link -> laser, at the documented mount point."""
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = BASE_FRAME
        t.child_frame_id = LASER_FRAME
        t.transform.translation.x = LIDAR_X
        t.transform.translation.y = LIDAR_Y
        t.transform.translation.z = LIDAR_Z
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0
        self.static_broadcaster.sendTransform(t)

    def on_scan(self, msg):
        """Republish with the frame slam_toolbox will resolve through TF."""
        msg.header.frame_id = LASER_FRAME
        self.scan_pub.publish(msg)
        self.n_scan += 1

    def on_odom(self, msg):
        """odom -> base_link, straight from the devkit's pose."""
        t = TransformStamped()
        t.header.stamp = msg.header.stamp
        t.header.frame_id = ODOM_FRAME
        t.child_frame_id = BASE_FRAME
        t.transform.translation.x = msg.pose.pose.position.x
        t.transform.translation.y = msg.pose.pose.position.y
        t.transform.translation.z = 0.0      # planar: keep the tree flat
        t.transform.rotation = msg.pose.pose.orientation
        self.tf_broadcaster.sendTransform(t)
        self.n_odom += 1

    def report(self):
        """Heartbeat, so a silent failure is visible rather than puzzling."""
        if self.n_scan == 0 and self.n_odom == 0:
            self.get_logger().warn(
                "nothing received. If 'ros2 topic hz "
                f"{TOPIC_SCAN_IN}' does show a rate in another terminal, "
                "the topics are reaching the host and the fault is here, "
                "not in discovery.")
        else:
            self.get_logger().info(
                f"scans {self.n_scan}, odom {self.n_odom} "
                f"(~{self.n_scan / 5.0:.1f} Hz, ~{self.n_odom / 5.0:.1f} Hz)")
        self.n_scan = 0
        self.n_odom = 0


def main():
    rclpy.init()
    node = SlamBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
