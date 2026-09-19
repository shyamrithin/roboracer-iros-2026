#!/usr/bin/env python3
# =============================================================================
# mpc_tracker.py
# RoboRacer IROS 2026 / CEM Navigators
# =============================================================================
#
# CODE DESCRIPTION
# -----------------------------------------------------------------------------
# Nonlinear model predictive control for following the stored raceline, as a
# drop-in alternative to raceline_tracker.py. Same map, same raceline, same
# localisation - the controller is the only thing that changes, so any
# difference in lap time is attributable to it.
#
# WHY TRY THIS
#   Pure pursuit is geometric and memoryless: it picks an aim point a fixed
#   distance ahead and solves for a steering angle, with no notion of what
#   happens after that. It cannot know a corner is coming, so it cannot set up
#   for one, and it cannot trade braking now against turning later.
#
#   Measured cross-track error on v20_r is 0.11-0.20 m at the first hairpin
#   and 0.04-0.20 m at the upper span, against a corridor that leaves roughly
#   0.8 m either side of the line. The tracker is leaving most of the
#   available room unused, and every attempt to convert that into speed by
#   changing the LINE has failed: minimum curvature cut the apexes, corner
#   widening raised the median radius but not the minimum, and local speed
#   increases collide. If the room is to be used, the controller has to use
#   it.
#
#   MPC optimises a sequence of inputs over a horizon against a vehicle model
#   and explicit constraints, so braking and turning are solved together
#   rather than approximated by hand-tuned a_dec. That coupling is exactly
#   what the friction-ellipse experiment showed the current profile gets
#   wrong.
#
# VEHICLE MODEL
#   Kinematic bicycle, which the slip probe validated directly: the measured
#   yaw-rate ratio against v*tan(delta)/L is 1.012 with p10 0.946 and p90
#   1.070, so the vehicle does not slide and no tyre model is warranted.
#
#       x'   = v cos(psi)
#       y'   = v sin(psi)
#       psi' = v tan(delta) / L
#       v'   = a
#
#   L = 0.324 m, |delta| <= 0.5236 rad. Throttle maps to speed through the
#   measured speed_per_throttle of 24.3, and the acceleration bounds come
#   from the coastdown (4.3 m/s2 of drag) and observed throttle response
#   (about 3.0 m/s2).
#
# HONEST EXPECTATIONS
#   IPOPT solving a 12-step nonlinear program in Python will manage perhaps
#   15-25 Hz, against pure pursuit's 50. The particle filter runs
#   independently at full rate so the pose stays fresh, but control updates
#   will be sparser. That may cost more than the optimisation gains - this is
#   an experiment, not a replacement, and raceline_tracker.py remains the
#   submitted controller until this beats 10.2 s over a long run.
#
# USAGE
#   export ROS_DOMAIN_ID=0
#   export FASTRTPS_DEFAULT_PROFILES_FILE=$HOME/roboracer/devkit_src/tools/fastdds_udp.xml
#   python3 mpc_tracker.py ~/mapdata/raceline_v20_r.csv
#
#   The particle filter must be running and converged first, exactly as for
#   raceline_tracker.py. The gap follower must NOT be running.
#
# DEPENDENCIES: rclpy, numpy, casadi, tf2_ros
# =============================================================================

import math
import os
import sys
import time

import numpy as np
import casadi as ca

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32
from tf2_ros import Buffer, TransformListener


WHEELBASE_M = 0.324
WHEEL_RADIUS_M = 0.0590
MAX_STEER_RAD = 0.5236
SPEED_PER_THROTTLE = 24.3


def load_raceline(path):
    rows = [l for l in open(path) if l.strip()]
    start = 1 if not rows[0].lstrip().startswith(('-', '0', '1', '2', '3',
                                                  '4', '5', '6', '7', '8',
                                                  '9')) else 0
    P = np.array([[float(v) for v in l.split(',')[:3]] for l in rows[start:]])
    return P[:, 0], P[:, 1], P[:, 2]


def build_solver(N, dt, a_min, a_max):
    """One-time construction of the NLP. Called at startup, not per cycle."""
    nx, nu = 4, 2
    X = ca.SX.sym('X', nx, N + 1)
    U = ca.SX.sym('U', nu, N)
    # parameters: initial state, then per-step reference (x, y, psi, v)
    P = ca.SX.sym('P', nx + N * 4)

    Q_pos = 40.0        # position error
    Q_psi = 2.0         # heading error
    Q_v = 1.2           # speed error
    R_d = 1.5           # steering magnitude
    R_a = 0.05          # acceleration magnitude
    Rd_d = 25.0         # steering RATE - the main smoothness term
    Rd_a = 0.5          # acceleration rate

    cost = 0
    g = [X[:, 0] - P[0:nx]]

    for k in range(N):
        xk, uk = X[:, k], U[:, k]
        ref = P[nx + 4 * k: nx + 4 * k + 4]

        dx = xk[0] - ref[0]
        dy = xk[1] - ref[1]
        dpsi = ca.atan2(ca.sin(xk[2] - ref[2]), ca.cos(xk[2] - ref[2]))
        dv = xk[3] - ref[3]

        cost += Q_pos * (dx * dx + dy * dy)
        cost += Q_psi * dpsi * dpsi
        cost += Q_v * dv * dv
        cost += R_d * uk[0] * uk[0] + R_a * uk[1] * uk[1]
        if k > 0:
            du = uk - U[:, k - 1]
            cost += Rd_d * du[0] * du[0] + Rd_a * du[1] * du[1]

        # RK2 integration of the kinematic bicycle
        def f(s, u):
            return ca.vertcat(s[3] * ca.cos(s[2]),
                              s[3] * ca.sin(s[2]),
                              s[3] * ca.tan(u[0]) / WHEELBASE_M,
                              u[1])
        k1 = f(xk, uk)
        k2 = f(xk + dt / 2 * k1, uk)
        g.append(X[:, k + 1] - (xk + dt * k2))

    # terminal position weight, so the horizon does not end carelessly
    refN = P[nx + 4 * (N - 1): nx + 4 * (N - 1) + 4]
    cost += 3.0 * Q_pos * ((X[0, N] - refN[0]) ** 2 + (X[1, N] - refN[1]) ** 2)

    opt = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
    nlp = {'x': opt, 'f': cost, 'g': ca.vertcat(*g), 'p': P}
    opts = {
        'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes',
        'ipopt.max_iter': 40, 'ipopt.tol': 1e-3, 'ipopt.acceptable_tol': 1e-2,
        'ipopt.warm_start_init_point': 'yes',
        'ipopt.mu_strategy': 'monotone',
    }
    solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

    nv = nx * (N + 1) + nu * N
    lbx = -np.inf * np.ones(nv)
    ubx = np.inf * np.ones(nv)
    # speed bounds on the state
    for k in range(N + 1):
        lbx[nx * k + 3] = 0.0
        ubx[nx * k + 3] = 8.0
    # control bounds
    off = nx * (N + 1)
    for k in range(N):
        lbx[off + nu * k] = -MAX_STEER_RAD
        ubx[off + nu * k] = MAX_STEER_RAD
        lbx[off + nu * k + 1] = a_min
        ubx[off + nu * k + 1] = a_max

    return solver, lbx, ubx, nx, nu, nv


class MpcTracker(Node):
    def __init__(self, path, N, dt):
        super().__init__('mpc_tracker')
        self.wx, self.wy, self.wv = load_raceline(path)
        self.n = len(self.wx)
        # reference heading at every waypoint
        dx = np.gradient(self.wx)
        dy = np.gradient(self.wy)
        self.wpsi = np.arctan2(dy, dx)
        print(f'loaded {self.n} waypoints, '
              f'speed {self.wv.min():.2f} to {self.wv.max():.2f} m/s')

        self.N, self.dt = N, dt
        self.solver, self.lbx, self.ubx, self.nx, self.nu, self.nv = \
            build_solver(N, dt, -4.3, 3.0)
        self.w0 = np.zeros(self.nv)
        self.lam_g0 = None
        self.idx = None
        self.warned = False
        self.solve_ms = []

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.steer_pub = self.create_publisher(
            Float32, '/autodrive/roboracer_1/steering_command', 10)
        self.throttle_pub = self.create_publisher(
            Float32, '/autodrive/roboracer_1/throttle_command', 10)
        self.enc_ref = {'left': None, 'right': None}
        self.enc_dist = {'left': 0.0, 'right': 0.0}
        self.prev_t = None
        self.centre_prev = 0.0
        self.speed = 0.0
        self.create_subscription(
            JointState, '/autodrive/roboracer_1/left_encoder',
            lambda m: self._enc(m, 'left'), qos)
        self.create_subscription(
            JointState, '/autodrive/roboracer_1/right_encoder',
            lambda m: self._enc(m, 'right'), qos)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_timer(dt, self.control)
        self.create_timer(2.0, self._report)
        self.last_log = 0.0
        self.get_logger().info(
            f'mpc_tracker ready. horizon {N} x {dt:.3f} s = {N*dt:.2f} s. '
            'The gap follower and raceline_tracker must not be running.')

    def _enc(self, msg, side):
        # Taken verbatim from raceline_tracker.py. Both encoders publish from
        # the same bridge cycle microseconds apart, so computing speed per
        # callback loses one wheel's increment to the dt guard and halves the
        # estimate. A reimplementation here produced 15 m/s spikes and zeros.
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

    def nearest(self, x, y):
        if self.idx is None:
            d = np.hypot(self.wx - x, self.wy - y)
            return int(np.argmin(d)), float(d.min())
        # Forward-only. With xte small a backward point never wins, but once
        # error grows past ~0.3 m one can, the index regresses, the horizon
        # reference marches from the wrong place and the solve locks up -
        # observed as wp 161 -> 156 and then stuck at 332 with xte 1.12.
        cand = [(self.idx + k) % self.n for k in range(0, 12)]
        d = np.hypot(self.wx[cand] - x, self.wy[cand] - y)
        j = int(np.argmin(d))
        return cand[j], float(d[j])

    def control(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time())
        except Exception:
            if not self.warned:
                self.get_logger().warn('waiting for map -> base_link')
                self.warned = True
            return
        self.warned = False

        x = tf.transform.translation.x
        y = tf.transform.translation.y
        q = tf.transform.rotation
        psi = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        v = max(self.speed, 0.0)

        i, dist = self.nearest(x, y)
        self.idx = i
        if dist > 1.5:
            self._publish(0.0, 0.0)
            self.get_logger().warn(
                f'nearest waypoint {dist:.2f} m away, stopping.',
                throttle_duration_sec=1.0)
            return

        # Reference: march along the line at the PLANNED speed, so the horizon
        # reference is a trajectory rather than a static set of points.
        ref = []
        j = i
        for k in range(self.N):
            # Step length fixed BEFORE marching. Re-reading wv[j] inside the
            # loop uses the entry speed all the way through a braking zone,
            # so the reference lands far beyond reach: measured span 5.51 m
            # against 2.32 m of travel, and the solver saturates steering
            # trying to get there.
            step = max(self.wv[j], 0.5) * self.dt
            acc = 0.0
            while acc < step:
                jn = (j + 1) % self.n
                acc += math.hypot(self.wx[jn] - self.wx[j],
                                  self.wy[jn] - self.wy[j])
                j = jn
            ref += [self.wx[j], self.wy[j], self.wpsi[j], self.wv[j]]

        self._ref_end = math.hypot(ref[-4] - x, ref[-3] - y)
        self._ref_span = sum(
            math.hypot(ref[4*k+4] - ref[4*k], ref[4*k+5] - ref[4*k+1])
            for k in range(self.N - 1))
        p = np.concatenate([[x, y, psi, v], np.asarray(ref)])
        t0 = time.time()
        try:
            args = dict(x0=self.w0, lbx=self.lbx, ubx=self.ubx,
                        lbg=0, ubg=0, p=p)
            if self.lam_g0 is not None:
                args['lam_g0'] = self.lam_g0
            sol = self.solver(**args)
            self.w0 = np.asarray(sol['x']).flatten()
            self.lam_g0 = np.asarray(sol['lam_g']).flatten()
        except Exception as e:
            self.get_logger().error(f'solver failed: {e}')
            return
        self.solve_ms.append((time.time() - t0) * 1000.0)

        off = self.nx * (self.N + 1)
        delta = float(self.w0[off])
        accel = float(self.w0[off + 1])

        steer = float(np.clip(delta / MAX_STEER_RAD, -1.0, 1.0))
        # This vehicle has no brake: negative throttle engages reverse, so
        # deceleration is commanded by cutting throttle to zero and letting
        # drivetrain drag do the work. Mapping a decelerating solution through
        # v_cmd/SPEED_PER_THROTTLE still yields positive throttle, so the
        # vehicle never slows - observed as v_m 5.52 against v_t 2.19 into the
        # first hairpin.
        # Use the same throttle mapping as raceline_tracker.py, which is
        # validated over 137 laps: feedforward on the target speed plus a
        # proportional term on the error. Deriving a new mapping from the
        # MPC's acceleration output was attempted three times and was wrong
        # each way - too blunt a cut oscillates, a multiplicative correction
        # compounds. The MPC's job here is the STEERING; the speed loop is
        # already solved.
        # Feedforward on the PLANNED speed, not on measured speed plus an
        # increment. The latter is positive feedback: when the vehicle is
        # already too fast, v_cmd follows it upward and the throttle command
        # chases rather than corrects. Observed running away to 15.27 m/s.
        v_target = float(self.wv[i])
        thr = 0.053 * v_target + 0.08 * (v_target - v)
        thr = float(np.clip(thr, 0.0, 0.6))

        self._publish(thr, steer)

        now = time.time()
        if now - self.last_log > 1.0:
            self.last_log = now
            xte = math.hypot(self.wx[i] - x, self.wy[i] - y)
            self.get_logger().info(
                f'wp={i:3d} xte={xte:.2f} v_t={self.wv[i]:.2f} v_m={v:.2f} '
                f'thr={thr:.3f} steer={steer:+.3f} '
                f'solve={np.mean(self.solve_ms[-40:]):.1f}ms '
                f'refend={self._ref_end:.2f} span={self._ref_span:.2f} '
                f'expect={v*self.N*self.dt:.2f}')

    def _publish(self, thr, steer):
        a, b = Float32(), Float32()
        a.data, b.data = float(thr), float(steer)
        self.throttle_pub.publish(a)
        self.steer_pub.publish(b)

    def _report(self):
        if len(self.solve_ms) > 20:
            a = np.asarray(self.solve_ms[-200:])
            self.get_logger().info(
                f'solve time median {np.median(a):.1f} ms '
                f'p90 {np.percentile(a, 90):.1f} ms '
                f'-> {1000.0/np.median(a):.0f} Hz achievable')


def main():
    if len(sys.argv) < 2:
        print('usage: python3 mpc_tracker.py <raceline.csv>')
        sys.exit(1)
    path = os.path.expanduser(sys.argv[1])
    if not os.path.isfile(path):
        print(f'no such file: {path}')
        sys.exit(1)

    rclpy.init()
    node = MpcTracker(path, N=14, dt=0.08)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node._publish(0.0, 0.0)
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
