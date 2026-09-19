#!/usr/bin/env python3
# =============================================================================
# slip_probe.py
# RoboRacer IROS 2026 / CEM Navigators
# =============================================================================
#
# CODE DESCRIPTION
# -----------------------------------------------------------------------------
# Measures whether the vehicle is sliding, by comparing the yaw rate the IMU
# actually reports against the yaw rate a no-slip kinematic bicycle model
# predicts from the commanded steering angle and the measured speed:
#
#     psi_dot_kinematic = v * tan(delta) / L
#
# If the two agree, the vehicle is doing exactly what the geometry asked and
# any cornering failure is a tracking or planning problem. If the measured
# rate is consistently LOWER than predicted, the front tyres are sliding -
# understeer - and the vehicle physically cannot follow the commanded arc, in
# which case no change to the racing line helps and only a lower entry speed
# does.
#
# WHY THIS MATTERS HERE
#   The minimum-curvature line (mc_r4 and its variants) clips repeatedly at
#   the upper span between the towers, waypoints 311-330, across roughly 220
#   laps. Shifting that section 8 cm in either direction failed: one way was
#   immediately worse, the other survived 37 laps and then cascaded the same
#   way. Lateral offset is evidently not the lever, which leaves two
#   possibilities that need opposite fixes - the vehicle is asking for a turn
#   it cannot execute, or it is executing correctly and the plan is wrong.
#
#   It also decides whether a tyre model is worth building. ForzaETH report
#   significantly faster lap times from extending pure pursuit with Pacejka
#   tyre formulas, but that only pays if there is slip to model. On a corner
#   that is radius-limited rather than grip-limited, the vehicle simply cannot
#   turn tighter and a dynamic model changes nothing.
#
# HOW TO READ THE OUTPUT
#   slip_ratio = measured_yaw_rate / kinematic_yaw_rate, reported where the
#   steering angle is large enough for the comparison to mean anything.
#
#     ~1.00        no slip; the kinematic model is accurate
#     0.85 - 0.95  mild understeer, normal near the limit
#     < 0.80       significant sliding; entry speed is too high for the grip
#     > 1.05       oversteer, or a sign or calibration error - check first
#
#   Watch the ratio against the section rather than the average. A ratio near
#   1.0 everywhere except one corner localises the problem precisely.
#
# SUBSCRIBES (all permissible at run time)
#   /autodrive/roboracer_1/imu              angular_velocity.z
#   /autodrive/roboracer_1/left_encoder     speed
#   /autodrive/roboracer_1/right_encoder    speed
#   /autodrive/roboracer_1/steering         commanded angle, feedback channel
#
# USAGE
#   Run alongside the normal filter and tracker, on the host:
#     python3 slip_probe.py
#     python3 slip_probe.py --csv ~/mapdata/slip.csv     to also log rows
#
# DEPENDENCIES: rclpy, numpy
# =============================================================================

import argparse
import math
import sys

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float32


WHEELBASE_M = 0.324
WHEEL_RADIUS_M = 0.0590
MAX_STEER_RAD = 0.5236          # check against the technical guide


class SlipProbe(Node):
    def __init__(self, csv_path, min_steer, window):
        super().__init__('slip_probe')
        self.min_steer = min_steer
        self.window = window

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=1)

        self.yaw_rate = None
        self.steer = None
        self.enc_ref = {'left': None, 'right': None}
        self.enc_dist = {'left': 0.0, 'right': 0.0}
        self.prev_dist = None
        self.prev_t = None
        self.speed = 0.0

        self.samples = []           # (speed, steer, measured, kinematic)
        self.fh = open(csv_path, 'w') if csv_path else None
        if self.fh:
            self.fh.write('t,v,steer_rad,yaw_meas,yaw_kin,ratio\n')

        self.create_subscription(
            Imu, '/autodrive/roboracer_1/imu', self._imu, qos)
        self.create_subscription(
            Float32, '/autodrive/roboracer_1/steering', self._steer, qos)
        self.create_subscription(
            JointState, '/autodrive/roboracer_1/left_encoder',
            lambda m: self._enc(m, 'left'), qos)
        self.create_subscription(
            JointState, '/autodrive/roboracer_1/right_encoder',
            lambda m: self._enc(m, 'right'), qos)
        self.create_timer(0.02, self._sample)
        self.create_timer(3.0, self._report)

        self.get_logger().info(
            'slip_probe: comparing IMU yaw rate against the kinematic '
            f'prediction, for |steer| > {min_steer:.3f} rad')

    def _imu(self, msg):
        self.yaw_rate = msg.angular_velocity.z

    def _steer(self, msg):
        # The feedback channel is in radians on this vehicle; the command
        # channel is normalised. Verify against the logged range before
        # trusting the numbers - see bag_analysis.py.
        self.steer = float(msg.data)

    def _enc(self, msg, side):
        if not msg.position:
            return
        pos = float(msg.position[0])
        if self.enc_ref[side] is None:
            self.enc_ref[side] = pos
            return
        self.enc_dist[side] = (pos - self.enc_ref[side]) * WHEEL_RADIUS_M
        if self.enc_ref['left'] is None or self.enc_ref['right'] is None:
            return
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        d = 0.5 * (self.enc_dist['left'] + self.enc_dist['right'])
        if self.prev_t is not None and t - self.prev_t > 0.05:
            self.speed = (d - self.prev_dist) / (t - self.prev_t)
            self.prev_t, self.prev_dist = t, d
        elif self.prev_t is None:
            self.prev_t, self.prev_dist = t, d

    def _sample(self):
        if self.yaw_rate is None or self.steer is None:
            return
        v = self.speed
        d = self.steer
        if abs(d) < self.min_steer or v < 0.8:
            return
        kin = v * math.tan(d) / WHEELBASE_M
        if abs(kin) < 1e-3:
            return
        ratio = self.yaw_rate / kin
        self.samples.append((v, d, self.yaw_rate, kin, ratio))
        if self.fh:
            t = self.get_clock().now().nanoseconds * 1e-9
            self.fh.write(f'{t:.3f},{v:.3f},{d:.4f},{self.yaw_rate:.4f},'
                          f'{kin:.4f},{ratio:.4f}\n')

    def _report(self):
        if len(self.samples) < 20:
            self.get_logger().info(
                f'{len(self.samples)} samples so far, need more cornering')
            return
        a = np.asarray(self.samples[-self.window:])
        v, d, meas, kin, ratio = a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4]
        # lateral acceleration, for context on where the limit sits
        alat = np.abs(meas) * v
        self.get_logger().info(
            f'n={len(a)}  ratio med {np.median(ratio):+.3f} '
            f'p10 {np.percentile(ratio, 10):+.3f} '
            f'p90 {np.percentile(ratio, 90):+.3f} | '
            f'|steer| med {np.median(np.abs(d)):.3f} rad | '
            f'a_lat med {np.median(alat):.2f} max {alat.max():.2f} m/s2')

    def summary(self):
        if len(self.samples) < 20:
            print('\nnot enough cornering samples')
            return
        a = np.asarray(self.samples)
        ratio = a[:, 4]
        alat = np.abs(a[:, 2]) * a[:, 0]
        print()
        print(f'{len(a)} cornering samples')
        print(f'  yaw-rate ratio  median {np.median(ratio):.3f}'
              f'  p10 {np.percentile(ratio, 10):.3f}'
              f'  p90 {np.percentile(ratio, 90):.3f}')
        print(f'  lateral accel   median {np.median(alat):.2f}'
              f'  p90 {np.percentile(alat, 90):.2f}'
              f'  max {alat.max():.2f} m/s2')
        m = np.median(ratio)
        if m > 1.05:
            print('  -> ratio above 1: check the steering units and sign '
                  'before drawing any conclusion')
        elif m > 0.95:
            print('  -> no meaningful slip. The kinematic model is accurate, '
                  'so cornering failures are tracking or planning problems '
                  'and a tyre model would add nothing.')
        elif m > 0.85:
            print('  -> mild understeer, normal near the limit. A tyre model '
                  'might be worth a few per cent.')
        else:
            print('  -> significant sliding. Entry speed exceeds the grip '
                  'available; lower a_lat rather than moving the line.')
        if self.fh:
            self.fh.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=None)
    ap.add_argument('--min-steer', type=float, default=0.12,
                    help='rad; below this the comparison is noise')
    ap.add_argument('--window', type=int, default=400)
    args, rest = ap.parse_known_args()

    rclpy.init(args=rest)
    node = SlipProbe(args.csv, args.min_steer, args.window)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.summary()
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
