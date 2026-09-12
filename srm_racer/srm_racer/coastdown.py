#!/usr/bin/env python3
# =============================================================================
# File        : coastdown.py
# Package     : srm_racer
# Project     : RoboRacer Sim Racing League @ IROS 2026 (AutoDRIVE Ecosystem)
# Target      : ROS 2 Humble / Python 3.10
# =============================================================================
# CODE DESCRIPTION
# -----------------------------------------------------------------------------
# Measures the deceleration this vehicle can actually achieve, which is the
# least known and most load-bearing parameter in the raceline velocity profile.
#
# WHY THIS IS NEEDED
# -----------------------------------------------------------------------------
# The velocity profile limits how fast speed may fall between waypoints:
#
#     v[i] <= sqrt(v[i+1]^2 + 2 * a_dec * ds)
#
# a_dec was assumed. It must be measured, because this vehicle has NO BRAKE -
# negative throttle engages reverse rather than braking, so deceleration comes
# from idle torque and tyre scrub alone. If the true figure is below the
# assumption, the profile will ask the vehicle to arrive at every corner
# faster than it can shed, which is precisely the failure mode that produced
# repeated exit collisions in earlier depth-throttle attempts.
#
# TRIGGERING (v2)
# -----------------------------------------------------------------------------
# v1 cut the throttle on a fixed timer. On a 7.5 s lap that fires wherever the
# vehicle happens to be, which is almost always mid-corner - the one place the
# measurement is invalid, and the place where coasting puts the vehicle
# nearest a wall. It produced collisions and almost no usable cuts.
#
# v2 triggers on geometry instead: the throttle is cut only when the steering
# is near zero AND the forward clearance is large, which together mean the
# vehicle is on a straight. A cooldown prevents repeated cuts within the same
# straight. If a corner arrives during a cut the window is abandoned early and
# throttle is restored, so the vehicle is never coasting into a turn.
#
# WHY NOT A CONVENTIONAL COASTDOWN
# -----------------------------------------------------------------------------
# A full coastdown from 5 m/s at even 1.5 m/s^2 needs over 8 m of straight.
# The longest straight on this circuit is about 6 m. So instead of one long
# decay, this node takes many short ones: it steers normally to stay on track
# and cuts the throttle to zero for a brief window every few seconds. The
# vehicle never slows enough to leave the corridor, and repeating the cut over
# many laps gives a distribution rather than a single reading.
#
# STEERING
# -----------------------------------------------------------------------------
# The disparity-extender steering is reproduced here rather than imported, so
# that nothing in the shipped gap_follower is touched or subclassed. Run this
# node INSTEAD of gap_follower - both publish throttle and would conflict.
#
# VALIDITY FILTERING
# -----------------------------------------------------------------------------
# A cut is only counted when the steering stayed small throughout. Tyre scrub
# at lock is itself a large decelerating force, so a cut taken mid-corner
# measures scrub rather than coasting drag and would overstate what is
# available on a straight. Cuts that span a corner are discarded.
#
# Deceleration is also reported against entry speed, since drag rises with
# speed and a single constant will understate braking from high speed and
# overstate it from low.
#
# OUTPUT
# -----------------------------------------------------------------------------
# A running summary every few cuts, and a final table on Ctrl-C giving mean,
# median and spread, plus the conservative value to use in the profile.
# =============================================================================

import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan, JointState
from std_msgs.msg import Float32

# --- Fixed vehicle geometry, do not tune -------------------------------------
LIDAR_TO_BUMPER_M = 0.141
CAR_HALF_WIDTH_M = 0.135
WHEELBASE_M = 0.324
MAX_STEER_RAD = 0.5236
WHEEL_RADIUS_M = 0.0590


class Coastdown(Node):
    """Disparity-extender steering with periodic throttle cuts."""

    def __init__(self):
        super().__init__('coastdown')

        # --- Steering, matched to the validated racing config ---------------
        self.declare_parameter('fov_deg', 60.0)
        self.declare_parameter('horizon_m', 8.0)
        self.declare_parameter('disparity_threshold_m', 0.20)
        self.declare_parameter('extend_margin_m', 0.16)
        self.declare_parameter('tie_tolerance_m', 0.50)
        self.declare_parameter('lookahead_m', 1.90)
        self.declare_parameter('steer_derate', 1.0)

        # --- Test schedule ---------------------------------------------------
        self.declare_parameter('cruise_throttle', 0.22)
        self.declare_parameter('coast_s', 0.6)
        self.declare_parameter('cooldown_s', 2.0)
        self.declare_parameter('trigger_max_steer', 0.10)
        self.declare_parameter('trigger_clear_m', 3.0)
        self.declare_parameter('abort_clear_m', 1.5)
        self.declare_parameter('max_steer_during_cut', 0.15)
        self.declare_parameter('min_entry_speed', 2.0)

        self._reload()

        # --- State ------------------------------------------------------------
        self.enc_ref = {'left': None, 'right': None}
        self.enc_dist = {'left': 0.0, 'right': 0.0}
        self.centre_prev = 0.0
        self.prev_centre_t = None
        self.speed = 0.0

        self.steer_norm = 0.0
        self.coasting = False
        self.phase_start = None
        self.samples = []          # (t, speed) during the current cut
        self.max_steer_seen = 0.0
        self.results = []          # (entry_speed, a_dec)
        self.discarded = 0

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.steer_pub = self.create_publisher(
            Float32, '/autodrive/roboracer_1/steering_command', 10)
        self.throttle_pub = self.create_publisher(
            Float32, '/autodrive/roboracer_1/throttle_command', 10)

        self.create_subscription(
            LaserScan, '/autodrive/roboracer_1/lidar', self.scan_callback, qos)
        self.create_subscription(
            JointState, '/autodrive/roboracer_1/left_encoder',
            lambda m: self.enc_callback(m, 'left'), qos)
        self.create_subscription(
            JointState, '/autodrive/roboracer_1/right_encoder',
            lambda m: self.enc_callback(m, 'right'), qos)

        self.last_cut_end = None
        self.aborted = 0

        self.get_logger().info(
            'coastdown v2 ready. Cuts throttle for {:.1f}s when |steer| < '
            '{:.2f} and clearance > {:.1f} m. Ctrl-C for the summary.'.format(
                self.coast_s, self.trigger_max_steer, self.trigger_clear_m))

    def _reload(self):
        g = self.get_parameter
        self.fov_rad = math.radians(g('fov_deg').value)
        self.horizon_m = g('horizon_m').value
        self.disparity_threshold_m = g('disparity_threshold_m').value
        self.extend_margin_m = g('extend_margin_m').value
        self.tie_tolerance_m = g('tie_tolerance_m').value
        self.lookahead_m = g('lookahead_m').value
        self.steer_derate = g('steer_derate').value
        self.cruise_throttle = g('cruise_throttle').value
        self.coast_s = g('coast_s').value
        self.cooldown_s = g('cooldown_s').value
        self.trigger_max_steer = g('trigger_max_steer').value
        self.trigger_clear_m = g('trigger_clear_m').value
        self.abort_clear_m = g('abort_clear_m').value
        self.max_steer_during_cut = g('max_steer_during_cut').value
        self.min_entry_speed = g('min_entry_speed').value

    # ---- Speed from averaged encoders --------------------------------------
    def enc_callback(self, msg: JointState, side: str):
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
        centre = 0.5 * (self.enc_dist['left'] + self.enc_dist['right'])

        if self.prev_centre_t is not None:
            dt = t - self.prev_centre_t
            if 1e-4 < dt < 0.5:
                raw = (centre - self.centre_prev) / dt
                if abs(raw) < 25.0:
                    # Light filtering only: heavy smoothing would blur the
                    # decay we are trying to measure.
                    self.speed += 0.5 * (raw - self.speed)
        self.centre_prev = centre
        self.prev_centre_t = t

        if self.coasting:
            self.samples.append((t, self.speed))

    # ---- Steering, reproduced from the racing stack -------------------------
    def scan_callback(self, msg: LaserScan):
        self._reload()
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        ranges, angles = self._prepare(msg)
        if ranges.size == 0:
            self._publish(0.0, 0.0)
            return
        self._extend(ranges, msg.angle_increment)
        best = self._select(ranges, angles)
        aim = float(angles[best])
        delta = math.atan2(2.0 * WHEELBASE_M * math.sin(aim), self.lookahead_m)
        self.steer_norm = float(np.clip(delta / MAX_STEER_RAD, -1.0, 1.0))

        # Forward clearance: nearest return in a narrow cone dead ahead.
        cone = np.abs(angles) <= math.radians(6.0)
        clear = float(np.min(ranges[cone])) if cone.any() else 0.0

        if self.coasting:
            self.max_steer_seen = max(self.max_steer_seen,
                                      abs(self.steer_norm))
            elapsed = t - self.phase_start

            # Abandon the window rather than coast into a corner.
            if clear < self.abort_clear_m:
                self.aborted += 1
                self.coasting = False
                self.last_cut_end = t
                thr = self.cruise_throttle * (
                    1.0 - self.steer_derate * abs(self.steer_norm))
                self._publish(self.steer_norm, max(thr, 0.0))
                return

            if elapsed >= self.coast_s:
                self._finish_cut()
                self.coasting = False
                self.last_cut_end = t
            self._publish(self.steer_norm, 0.0)
            return

        ready = (self.last_cut_end is None
                 or t - self.last_cut_end >= self.cooldown_s)
        on_straight = (abs(self.steer_norm) < self.trigger_max_steer
                       and clear > self.trigger_clear_m)

        if ready and on_straight and self.speed >= self.min_entry_speed:
            self.coasting = True
            self.phase_start = t
            self.samples = []
            self.max_steer_seen = abs(self.steer_norm)
            self._publish(self.steer_norm, 0.0)
            return

        thr = self.cruise_throttle * (
            1.0 - self.steer_derate * abs(self.steer_norm))
        self._publish(self.steer_norm, max(thr, 0.0))

    def _finish_cut(self):
        """Fit the decay and keep it only if the vehicle stayed near straight."""
        if len(self.samples) < 5:
            self.discarded += 1
            return
        t = np.array([s[0] for s in self.samples])
        v = np.array([s[1] for s in self.samples])
        entry = v[0]

        if self.max_steer_seen > self.max_steer_during_cut:
            self.discarded += 1
            return
        if entry < self.min_entry_speed:
            self.discarded += 1
            return

        slope = np.polyfit(t - t[0], v, 1)[0]
        if slope >= 0.0:
            self.discarded += 1
            return

        a = -slope
        self.results.append((entry, a))
        self.get_logger().info(
            'cut {:2d}: entry {:.2f} m/s -> {:.2f} m/s over {:.2f}s, '
            'a_dec = {:.2f} m/s^2  (max steer {:.2f})'.format(
                len(self.results), entry, v[-1], t[-1] - t[0], a,
                self.max_steer_seen))

        if len(self.results) % 5 == 0:
            self.summary(brief=True)

    def summary(self, brief=False):
        if not self.results:
            print('\nno valid cuts. Lower min_entry_speed, or raise '
                  'max_steer_during_cut if the circuit has no straight '
                  'long enough.')
            return
        a = np.array([r[1] for r in self.results])
        e = np.array([r[0] for r in self.results])
        print()
        print('=' * 60)
        print(f'  valid cuts {len(a)}, discarded {self.discarded}, '
              f'aborted into corner {self.aborted}')
        print(f'  a_dec  mean {a.mean():.2f}  median {np.median(a):.2f}  '
              f'min {a.min():.2f}  max {a.max():.2f}  std {a.std():.2f}')
        print(f'  entry speeds {e.min():.2f} to {e.max():.2f} m/s')
        if not brief and len(a) >= 6:
            lo = a[e < np.median(e)]
            hi = a[e >= np.median(e)]
            print(f'  below median entry speed: a_dec {lo.mean():.2f}')
            print(f'  above median entry speed: a_dec {hi.mean():.2f}')
            print('  (higher at speed means drag dominates, which a single '
                  'constant\n   will understate when braking from high speed)')
            print()
            print(f'  USE {np.percentile(a, 25):.2f} m/s^2 in the velocity '
                  'profile.')
            print('  That is the 25th percentile: the profile must hold on a '
                  'bad lap,\n  not an average one.')
        print('=' * 60)

    # ---- Steering internals -------------------------------------------------
    def _prepare(self, msg):
        r = np.asarray(msg.ranges, dtype=np.float64)
        r = np.nan_to_num(r, nan=0.0, posinf=self.horizon_m, neginf=0.0)
        r = np.maximum(r - LIDAR_TO_BUMPER_M, 0.0)
        r = np.minimum(r, self.horizon_m)
        angles = msg.angle_min + np.arange(r.size) * msg.angle_increment
        keep = np.abs(angles) <= self.fov_rad
        return r[keep], angles[keep]

    def _extend(self, ranges, inc):
        half = CAR_HALF_WIDTH_M + self.extend_margin_m
        diffs = np.diff(ranges)
        for i in np.flatnonzero(np.abs(diffs) > self.disparity_threshold_m):
            if diffs[i] > 0:
                idx, d, step = i, ranges[i], 1
            else:
                idx, d, step = i + 1, ranges[i + 1], -1
            if d < 1e-3:
                continue
            n = int(math.atan2(half, d) / inc) + 1
            if step > 0:
                lo, hi = idx, min(idx + n + 1, ranges.size)
            else:
                lo, hi = max(idx - n, 0), idx + 1
            ranges[lo:hi] = np.minimum(ranges[lo:hi], d)

    def _select(self, ranges, angles):
        if self.tie_tolerance_m <= 0.0:
            return int(np.argmax(ranges))
        peak = float(np.max(ranges))
        near = np.flatnonzero(ranges >= peak - self.tie_tolerance_m)
        return int(near[np.argmin(np.abs(angles[near]))])

    def _publish(self, steer, throttle):
        m = Float32()
        m.data = float(np.clip(steer, -1.0, 1.0))
        self.steer_pub.publish(m)
        m2 = Float32()
        m2.data = float(np.clip(throttle, 0.0, 1.0))
        self.throttle_pub.publish(m2)


def main(args=None):
    rclpy.init(args=args)
    node = Coastdown()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.summary()
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
