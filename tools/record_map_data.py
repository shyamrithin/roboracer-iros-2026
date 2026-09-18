#!/usr/bin/env python3
# =============================================================================
# record_map_data.py
# RoboRacer IROS 2026 / CEM Navigators
# =============================================================================
#
# CODE DESCRIPTION
# -----------------------------------------------------------------------------
# Records synchronised LiDAR scans and ground-truth pose to an .npz file, for
# OFFLINE construction of an occupancy grid of the Phase 2 competition track.
#
# Subscribes to:
#     /autodrive/roboracer_1/lidar   sensor_msgs/LaserScan
#     /autodrive/roboracer_1/ips     geometry_msgs/Point      (position)
#     /autodrive/roboracer_1/imu     sensor_msgs/Imu          (heading)
#
# THIS IS AN OFFLINE TOOL. IT MUST NEVER BE PART OF THE SUBMITTED CONTAINER.
# ---------------------------------------------------------------------------
# It reads simulation ground truth. Rules section 2.4 prohibits utilizing
# ground truth data; the organisers confirmed on Slack (2026-09-17) that using
# IPS offline, before the race, purely to build a map is permitted, provided
# the racing stack subscribes only to /lidar at run time and localises against
# the saved map. Keep this file in tools/, never in srm_racer/.
#
# WHY GROUND TRUTH IS NEEDED HERE
#   slam_toolbox does not converge on this layout. Two near-identical towers
#   and two parallel straights make the scan matcher register against the
#   wrong section: map->odom swings by metres and tens of degrees per lap and
#   the saved map comes out as several rotated copies. Three configurations
#   were tried, including a crawling run at the full 56 Hz scan rate.
#
#   A screenshot-derived map was built as an alternative (build_track_map.py)
#   and is geometrically usable but not trustworthy in detail: the simulator
#   view is a perspective projection, the wall threshold has to be fitted, and
#   the resulting corridor half-widths are inflated at the hairpins, which is
#   what makes the TUM optimiser's spline normals cross.
#
#   With exact poses there is no scan matching, no ambiguity, no thresholding
#   and no scale transfer. Every scan lands where it actually was.
#
# HOW TO DRIVE THE RECORDING
#   Drive SLOWLY and take the full width of the corridor where you can. The
#   map is built from what the LiDAR saw, so walls are only mapped where a
#   beam reached them. Two or three laps is plenty; more adds little.
#
# USAGE
#   export ROS_DOMAIN_ID=0
#   export FASTRTPS_DEFAULT_PROFILES_FILE=$HOME/roboracer/devkit_src/tools/fastdds_udp.xml
#   python3 record_map_data.py -o ~/mapdata/bridge_scans.npz
#   # Ctrl-C when done; the file is written on shutdown.
#
# DEPENDENCIES: rclpy, numpy
# =============================================================================

import argparse
import sys

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan, Imu
from geometry_msgs.msg import Point


class MapRecorder(Node):
    def __init__(self, out_path):
        super().__init__('map_recorder')
        self.out_path = out_path

        # The simulator publishes sensor data best-effort. Using the default
        # reliable profile here silently receives nothing.
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=10)

        self.pose = None          # (x, y) most recent IPS
        self.yaw = None           # most recent IMU yaw
        self.scans = []           # one row of ranges per accepted scan
        self.poses = []           # matching (x, y, yaw)
        self.meta = None          # (angle_min, angle_increment, range_max)
        self.skipped = 0

        self.create_subscription(
            Point, '/autodrive/roboracer_1/ips', self._on_ips, qos)
        self.create_subscription(
            Imu, '/autodrive/roboracer_1/imu', self._on_imu, qos)
        self.create_subscription(
            LaserScan, '/autodrive/roboracer_1/lidar', self._on_scan, qos)
        self.create_timer(2.0, self._report)

        self.get_logger().info(f'recording to {out_path}; Ctrl-C to finish')

    def _on_ips(self, msg):
        self.pose = (msg.x, msg.y)

    def _on_imu(self, msg):
        q = msg.orientation
        # yaw from quaternion, z-up
        self.yaw = np.arctan2(2.0 * (q.w * q.z + q.x * q.y),
                              1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    def _on_scan(self, msg):
        if self.pose is None or self.yaw is None:
            self.skipped += 1
            return
        if self.meta is None:
            self.meta = (msg.angle_min, msg.angle_increment, msg.range_max)
            self.get_logger().info(
                f'scan: {len(msg.ranges)} beams, '
                f'angle_min {msg.angle_min:.4f}, '
                f'increment {msg.angle_increment:.6f}, '
                f'range_max {msg.range_max:.2f}')
        self.scans.append(np.asarray(msg.ranges, dtype=np.float32))
        self.poses.append((self.pose[0], self.pose[1], self.yaw))

    def _report(self):
        p = self.poses[-1] if self.poses else (0.0, 0.0, 0.0)
        self.get_logger().info(
            f'scans {len(self.scans)}  skipped {self.skipped}  '
            f'pose ({p[0]:+.2f}, {p[1]:+.2f}) yaw {p[2]:+.3f}')

    def save(self):
        if not self.scans:
            self.get_logger().error('nothing recorded')
            return
        np.savez_compressed(
            self.out_path,
            ranges=np.stack(self.scans),
            poses=np.asarray(self.poses, dtype=np.float64),
            angle_min=self.meta[0],
            angle_increment=self.meta[1],
            range_max=self.meta[2])
        xs = np.asarray(self.poses)[:, 0]
        ys = np.asarray(self.poses)[:, 1]
        print()
        print(f'wrote {self.out_path}')
        print(f'  {len(self.scans)} scans')
        print(f'  x range {xs.min():+.2f} .. {xs.max():+.2f} '
              f'({xs.max()-xs.min():.2f} m)')
        print(f'  y range {ys.min():+.2f} .. {ys.max():+.2f} '
              f'({ys.max()-ys.min():.2f} m)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-o', '--out', default='bridge_scans.npz')
    args = ap.parse_args()

    rclpy.init()
    node = MapRecorder(args.out)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save()
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
