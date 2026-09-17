#!/usr/bin/env python3
# =============================================================================
# particle_filter.py
# RoboRacer IROS 2026 / Team 26 CEM Navigators
# =============================================================================
#
# CODE DESCRIPTION
# -----------------------------------------------------------------------------
# Monte Carlo localisation against a prebuilt occupancy grid, using LiDAR,
# wheel encoders and IMU only. Publishes map -> odom -> base_link so that
# raceline_tracker.py runs unmodified.
#
# THIS IS THE RACE-LEGAL REPLACEMENT FOR tools/ips_tf_shim.py.
#   Subscribes ONLY to:
#       /autodrive/roboracer_1/lidar          sensor_msgs/LaserScan
#       /autodrive/roboracer_1/left_encoder   sensor_msgs/JointState
#       /autodrive/roboracer_1/right_encoder  sensor_msgs/JointState
#       /autodrive/roboracer_1/imu            sensor_msgs/Imu
#   It does NOT read /ips, /odom or /tf. Rules section 2.4 prohibits
#   utilizing simulation ground truth; the map it localises against was built
#   offline, which the organisers confirmed is permitted (Slack, 2026-09-17).
#
# WHY A PARTICLE FILTER AND NOT SCAN MATCHING
#   slam_toolbox does not converge on this track. Two near-identical towers
#   and two parallel straights make a scan matcher register against the wrong
#   section; map->odom swung by metres and tens of degrees per lap across
#   three different configurations. A particle filter carries many hypotheses
#   at once and lets motion evidence eliminate the wrong ones, instead of
#   committing to a single best match every scan.
#
#   It also starts from a KNOWN pose. The vehicle begins every run at
#   (0.80, 3.16) yaw -1.571, measured repeatedly across resets. Tracking from
#   a known start is a far easier problem than global localisation, and on a
#   symmetric track that difference is the whole ballgame. If the filter is
#   ever asked to relocalise from scratch - after the collision reset the
#   rules describe - it may well pick the wrong tower. Do not collide.
#
# SENSOR MODEL
#   Likelihood field, as in Thrun et al. and as AMCL uses by default. The
#   Euclidean distance transform of the occupied cells is precomputed once, so
#   scoring a beam is a single array lookup rather than a ray march. For 800
#   particles and 60 beams that is 48000 lookups per scan, which numpy handles
#   in about a millisecond; ray marching the same set would be 100x that and
#   would not fit in the 18 ms budget at 56 Hz.
#
#   Weights are computed in log space and normalised by the maximum before
#   exponentiating, so a confident filter does not underflow to all-zero.
#
# MOTION MODEL
#   Distance from the wheel encoders, heading from the IMU. Both are legal
#   inputs. Noise is added proportional to the distance travelled, plus a
#   small floor so the filter never collapses to zero spread while stationary.
#
# USAGE
#   export ROS_DOMAIN_ID=0
#   export FASTRTPS_DEFAULT_PROFILES_FILE=$HOME/roboracer/devkit_src/tools/fastdds_udp.xml
#   python3 particle_filter.py ~/mapdata/bridge_ips_cl.yaml
#
#   Then the tracker, exactly as with the shim:
#     python3 raceline_tracker.py ~/mapdata/raceline_bridge_v2.csv \
#       --ros-args -p lookahead_max_m:=1.1 -p lookahead_min_m:=0.6 \
#                  -p lookahead_base_m:=0.5
#
# CHECK BEFORE TRUSTING IT
#   Run it alongside ips_tf_shim.py disabled and compare the reported pose
#   against the IPS value printed by the shim. They should agree to a few
#   centimetres. That comparison is a diagnostic, not part of the race stack.
#
# DEPENDENCIES: rclpy, numpy, scipy, Pillow, tf2_ros
# =============================================================================

import argparse
import math
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import LaserScan, Imu, JointState
from tf2_ros import TransformBroadcaster


WHEEL_RADIUS_M = 0.0590      # check against raceline_tracker.py before racing


def load_map(path):
    cfg = {}
    for line in open(path):
        line = line.strip()
        if not line or line.startswith('#') or ':' not in line:
            continue
        k, v = line.split(':', 1)
        cfg[k.strip()] = v.strip()
    base = path.rsplit('/', 1)[0] if '/' in path else '.'
    grid = np.array(Image.open(f"{base}/{cfg['image']}"))
    res = float(cfg['resolution'])
    org = [float(x) for x in cfg['origin'].strip('[]').split(',')[:2]]
    grid = np.flipud(grid)                      # row 0 becomes y = origin
    free = grid == 254
    # distance in metres from every cell to the nearest non-free cell
    dist = ndimage.distance_transform_edt(free) * res
    return free, dist, res, org


class ParticleFilter(Node):
    def __init__(self, map_yaml, n_particles, n_beams):
        super().__init__('particle_filter')
        self.free, self.dist, self.res, self.org = load_map(map_yaml)
        self.H, self.W = self.dist.shape
        self.get_logger().info(
            f'map {self.W} x {self.H} @ {self.res} m, '
            f'origin ({self.org[0]:.3f}, {self.org[1]:.3f})')

        self.N = n_particles
        self.n_beams = n_beams
        self.sigma_hit = 0.15
        self.alpha_dist = 0.08
        self.alpha_yaw = 0.02         # heading noise per metre
        self.min_noise = 0.0005

        # start pose, measured across resets
        self.declare_parameter('init_x', 0.80)
        self.declare_parameter('init_y', 3.16)
        self.declare_parameter('init_yaw', -1.571)
        self.declare_parameter('init_spread_m', 0.10)
        self.declare_parameter('init_spread_rad', 0.05)
        ix = self.get_parameter('init_x').value
        iy = self.get_parameter('init_y').value
        iyaw = self.get_parameter('init_yaw').value
        sm = self.get_parameter('init_spread_m').value
        sr = self.get_parameter('init_spread_rad').value

        rng = np.random.default_rng(0)
        self.P = np.empty((self.N, 3))
        self.P[:, 0] = ix + rng.normal(0, sm, self.N)
        self.P[:, 1] = iy + rng.normal(0, sm, self.N)
        self.P[:, 2] = iyaw + rng.normal(0, sr, self.N)
        self.w = np.full(self.N, 1.0 / self.N)
        self.rng = rng

        self.enc_ref = {'left': None, 'right': None}
        self.enc_dist = {'left': 0.0, 'right': 0.0}
        self.last_dist = None
        self.yaw = None
        self.yaw_ref = None
        self.beam_ang = None
        self.updates = 0
        self.est = (ix, iy, iyaw)

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(
            JointState, '/autodrive/roboracer_1/left_encoder',
            lambda m: self._enc(m, 'left'), qos)
        self.create_subscription(
            JointState, '/autodrive/roboracer_1/right_encoder',
            lambda m: self._enc(m, 'right'), qos)
        self.create_subscription(
            Imu, '/autodrive/roboracer_1/imu', self._imu, qos)
        self.create_subscription(
            LaserScan, '/autodrive/roboracer_1/lidar', self._scan, qos)

        self.br = TransformBroadcaster(self)
        self.create_timer(0.01, self._broadcast)
        self.create_timer(2.0, self._report)
        self.get_logger().info(
            f'{self.N} particles, {self.n_beams} beams. '
            f'LiDAR + encoders + IMU only, no ground truth.')

    # ---- sensors ----------------------------------------------------------
    def _enc(self, msg, side):
        if not msg.position:
            return
        pos = float(msg.position[0])
        if self.enc_ref[side] is None:
            self.enc_ref[side] = pos
            return
        self.enc_dist[side] = (pos - self.enc_ref[side]) * WHEEL_RADIUS_M

    def _imu(self, msg):
        q = msg.orientation
        self.yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                              1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    def _scan(self, msg):
        if self.yaw is None or self.enc_ref['left'] is None:
            return
        if self.beam_ang is None:
            n = len(msg.ranges)
            self.idx = np.linspace(0, n - 1, self.n_beams).astype(int)
            self.beam_ang = (msg.angle_min
                             + msg.angle_increment * self.idx)
            self.range_max = msg.range_max

        travelled = 0.5 * (self.enc_dist['left'] + self.enc_dist['right'])
        if self.last_dist is None:
            self.last_dist = travelled
            self.yaw_ref = self.yaw
            return
        ds = travelled - self.last_dist
        self.last_dist = travelled

        # ---- motion ----
        # Heading comes from the IMU directly; the encoders give distance.
        dyaw = math.atan2(math.sin(self.yaw - self.yaw_ref),
                          math.cos(self.yaw - self.yaw_ref))
        self.yaw_ref = self.yaw
        mag = abs(ds)
        sd = self.alpha_dist * mag + self.min_noise
        sy = self.alpha_yaw * mag + self.min_noise
        # The IMU gives absolute heading directly and is far more reliable
        # than anything the filter can infer, so pull every particle back
        # towards it. Without this, resampling near the towers can flip the
        # whole cloud onto a yaw hypothesis 40 deg out - seen at wp 280-307.
        self.P[:, 2] += dyaw + self.rng.normal(0, sy, self.N)
        err = np.arctan2(np.sin(self.yaw - self.P[:, 2]),
                         np.cos(self.yaw - self.P[:, 2]))
        self.P[:, 2] += 0.5 * err
        step = ds + self.rng.normal(0, sd, self.N)
        self.P[:, 0] += step * np.cos(self.P[:, 2])
        self.P[:, 1] += step * np.sin(self.P[:, 2])

        # ---- correction ----
        r = np.asarray(msg.ranges, dtype=np.float64)[self.idx]
        good = np.isfinite(r) & (r > 0.15) & (r < self.range_max * 0.95)
        if good.sum() < 8:
            return
        rr = r[good]
        ba = self.beam_ang[good]

        # endpoints of every beam for every particle
        ang = self.P[:, 2][:, None] + ba[None, :]
        ex = self.P[:, 0][:, None] + rr[None, :] * np.cos(ang)
        ey = self.P[:, 1][:, None] + rr[None, :] * np.sin(ang)

        cx = ((ex - self.org[0]) / self.res).astype(np.int32)
        cy = ((ey - self.org[1]) / self.res).astype(np.int32)
        inside = (cx >= 0) & (cx < self.W) & (cy >= 0) & (cy < self.H)
        np.clip(cx, 0, self.W - 1, out=cx)
        np.clip(cy, 0, self.H - 1, out=cy)

        d = self.dist[cy, cx]
        # a beam landing outside the map is as bad as landing far from a wall
        d = np.where(inside, d, 2.0)
        logw = -(d * d).sum(axis=1) / (2.0 * self.sigma_hit ** 2)

        logw -= logw.max()
        w = np.exp(logw)
        s = w.sum()
        if s <= 0 or not np.isfinite(s):
            return
        self.w = w / s

        # ---- resample when the effective sample size drops ----
        ess = 1.0 / np.sum(self.w ** 2)
        if ess < self.N * 0.15:
            pos = (self.rng.random() + np.arange(self.N)) / self.N
            idx = np.searchsorted(np.cumsum(self.w), pos)
            self.P = self.P[np.clip(idx, 0, self.N - 1)].copy()
            # Roughening: without it, resampling at a feature-rich corner
            # collapses the cloud to millimetres and leaves no diversity to
            # recover with on the next straight.
            self.P[:, 0] += self.rng.normal(0, 0.03, self.N)
            self.P[:, 1] += self.rng.normal(0, 0.03, self.N)
            self.P[:, 2] += self.rng.normal(0, 0.01, self.N)
            self.w = np.full(self.N, 1.0 / self.N)

        # ---- estimate: weighted mean, circular for heading ----
        x = float(np.sum(self.w * self.P[:, 0]))
        y = float(np.sum(self.w * self.P[:, 1]))
        yaw = math.atan2(float(np.sum(self.w * np.sin(self.P[:, 2]))),
                         float(np.sum(self.w * np.cos(self.P[:, 2]))))
        self.est = (x, y, yaw)
        self.ess = ess
        self.updates += 1

    # ---- output -----------------------------------------------------------
    def _broadcast(self):
        if self.beam_ang is None:
            return
        now = self.get_clock().now().to_msg()
        x, y, yaw = self.est

        t1 = TransformStamped()
        t1.header.stamp = now
        t1.header.frame_id = 'map'
        t1.child_frame_id = 'odom'
        t1.transform.rotation.w = 1.0

        t2 = TransformStamped()
        t2.header.stamp = now
        t2.header.frame_id = 'odom'
        t2.child_frame_id = 'base_link'
        t2.transform.translation.x = x
        t2.transform.translation.y = y
        t2.transform.rotation.z = math.sin(yaw * 0.5)
        t2.transform.rotation.w = math.cos(yaw * 0.5)
        self.br.sendTransform([t1, t2])

    def _report(self):
        if self.updates == 0:
            self.get_logger().warn('no scan updates yet')
            return
        x, y, yaw = self.est
        spread = float(np.sqrt(np.var(self.P[:, 0]) + np.var(self.P[:, 1])))
        self.get_logger().info(
            f'upd {self.updates}  pose ({x:+.2f}, {y:+.2f}) yaw {yaw:+.3f}  '
            f'spread {spread:.3f} m  ess {getattr(self, "ess", 0):.0f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('map_yaml')
    ap.add_argument('-n', '--particles', type=int, default=800)
    ap.add_argument('-b', '--beams', type=int, default=60)
    args, rest = ap.parse_known_args()

    rclpy.init(args=rest)
    node = ParticleFilter(args.map_yaml, args.particles, args.beams)
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
