#!/usr/bin/env python3
# =============================================================================
# raceline_viz.py
# -----------------------------------------------------------------------------
# Project : RoboRacer Sim Racing League @ IROS 2026 - Team CEM Navigators (26)
# Author  : Shyam Rithin
# Location: ~/roboracer/tools/raceline_viz.py
# Usage   : python3 raceline_viz.py ~/mapdata/raceline.csv
# =============================================================================
#
# DESCRIPTION
# -----------
# Publishes the generated raceline for inspection in rviz, so the line can be
# checked against the map BEFORE any controller is asked to drive it.
#
# This is the validation step the generator asks for. The waypoints are
# produced in world coordinates and transformed into the map frame using a
# constant measured with the vehicle stationary. If that constant is wrong,
# the line will sit outside the corridor, and a tracker following it would
# steer into a wall on the first lap. Looking at it costs a minute.
#
# WHAT TO CHECK IN RVIZ
#   * The green path lies inside the mapped corridor everywhere, with roughly
#     even margin either side.
#   * It closes on itself at the start/finish.
#   * The coloured spheres run red through the tight sections and green down
#     the straight - if the colours are inverted, the velocity profile is
#     reversed relative to the geometry.
#
# TOPICS (both latched, so rviz can be started afterwards)
#   /raceline        nav_msgs/Path          the line itself
#   /raceline_speed  visualization_msgs/MarkerArray   one sphere per waypoint,
#                                                     coloured by target speed
#
# Run on the HOST alongside slam_toolbox, with the same ROS_DOMAIN_ID and
# FASTRTPS_DEFAULT_PROFILES_FILE exports as everything else.
# =============================================================================

import sys
import os

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, HistoryPolicy

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray


def load(path):
    """Read the generator's CSV. Returns lists of x, y, v."""
    xs, ys, vs = [], [], []
    with open(path) as fh:
        header = fh.readline()
        if "x_map" not in header:
            print("warning: unexpected header, expected x_map,y_map,v_target,...")
        for line in fh:
            line = line.strip()
            if not line:
                continue
            p = line.split(",")
            xs.append(float(p[0]))
            ys.append(float(p[1]))
            vs.append(float(p[2]))
    return xs, ys, vs


class RacelineViz(Node):

    def __init__(self, xs, ys, vs, frame):
        super().__init__("raceline_viz")

        # Transient local so rviz picks it up whenever it starts.
        qos = QoSProfile(
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.path_pub = self.create_publisher(Path, "/raceline", qos)
        self.mark_pub = self.create_publisher(
            MarkerArray, "/raceline_speed", qos)

        self.xs, self.ys, self.vs, self.frame = xs, ys, vs, frame
        self.publish()
        self.create_timer(2.0, self.publish)

        vmin, vmax = min(vs), max(vs)
        self.get_logger().info(
            f"publishing {len(xs)} waypoints in frame '{frame}'. "
            f"speed {vmin:.2f} to {vmax:.2f} m/s. "
            "Add Path /raceline and MarkerArray /raceline_speed in rviz.")

    def publish(self):
        now = self.get_clock().now().to_msg()

        path = Path()
        path.header.stamp = now
        path.header.frame_id = self.frame
        for x, y in zip(self.xs, self.ys):
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x = x
            ps.pose.position.y = y
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)
        # Close the loop so the join at start/finish is visible.
        if path.poses:
            path.poses.append(path.poses[0])
        self.path_pub.publish(path)

        vmin, vmax = min(self.vs), max(self.vs)
        span = max(vmax - vmin, 1e-6)
        arr = MarkerArray()
        for i, (x, y, v) in enumerate(zip(self.xs, self.ys, self.vs)):
            m = Marker()
            m.header.stamp = now
            m.header.frame_id = self.frame
            m.ns = "raceline_speed"
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = x
            m.pose.position.y = y
            m.pose.position.z = 0.05
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.07
            f = (v - vmin) / span          # 0 slow, 1 fast
            m.color.r = float(1.0 - f)
            m.color.g = float(f)
            m.color.b = 0.0
            m.color.a = 1.0
            arr.markers.append(m)
        self.mark_pub.publish(arr)


def main():
    if len(sys.argv) < 2:
        print("usage: python3 raceline_viz.py <raceline.csv> [frame]")
        sys.exit(1)
    path = os.path.expanduser(sys.argv[1])
    if not os.path.isfile(path):
        print(f"error: no such file: {path}")
        sys.exit(1)
    frame = sys.argv[2] if len(sys.argv) > 2 else "map"

    xs, ys, vs = load(path)
    print(f"loaded {len(xs)} waypoints")
    print(f"  x [{min(xs):.2f},{max(xs):.2f}]  y [{min(ys):.2f},{max(ys):.2f}]")

    rclpy.init()
    node = RacelineViz(xs, ys, vs, frame)
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
