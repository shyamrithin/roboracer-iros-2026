#!/usr/bin/env python3
# =============================================================================
# File        : gap_follower.py
# Package     : srm_racer
# Project     : RoboRacer Sim Racing League @ IROS 2026 (AutoDRIVE Ecosystem)
# Target      : ROS 2 Humble / Python 3.10
# Created     : 2026-08-31
# Revised     : 2026-09-04  (v7: clean canonical disparity extender)
# Revised     : 2026-09-04  (v8: pure pursuit steering law, central tie-break
#                            among near-maximal headings)
# Revised     : 2026-09-15  (v9: Phase 2 competition track defaults; six
#                            parameter changes, no algorithmic change)
# Revised     : 2026-09-15  (v10: selectable depth source and speed law)
# =============================================================================
# CODE DESCRIPTION
# -----------------------------------------------------------------------------
# Canonical disparity extender for the RoboRacer digital twin in the AutoDRIVE
# Simulator. Consumes only the 2D LiDAR stream, which is a permissible input at
# run time, and emits normalised steering and throttle commands. No map, no
# stored knowledge of the circuit, and no ground-truth pose, so the behaviour
# transfers unchanged to an unseen racetrack. The core method is due to
# Otterness (UNC-Chapel Hill, 2019).
#
# REVISION NOTES (v10)
# -----------------------------------------------------------------------------
# DEFAULT BEHAVIOUR IS UNCHANGED FROM v9. depth_law defaults to 'off', which
# takes the same constant-throttle path v9 took. Everything below activates
# only when depth_law is set to 'ramp' or 'sqrt'.
#
# WHY
#   v9 runs a constant 0.10 throttle everywhere. On the Phase 2 track the
#   bridge deck logs dc at 5.5-6.5 m of clear road for roughly a quarter of a
#   47.8 m lap, all of it at 10 per cent throttle in a vehicle that held
#   5.4 m/s on 22 per cent. The straights are where the lap time is.
#
# 1. depth_source  (d1 | d2 | dc | da, default dc)
#    _path_depth previously returned d1 unconditionally. The four-way
#    comparison run measured d1 at +0.193 correlation with speed - INVERTED,
#    reading deeper when the vehicle is slower - with 23.2 per cent of samples
#    pinned at the 8.00 m horizon and 15.8 per cent below 0.3 m. Feeding that
#    to any speed law would accelerate INTO corners. dc, the narrow forward
#    wedge, measured -0.727 with no saturation at either end and a median of
#    3.38 m on the straight against 1.33 m in corners. On the Phase 2 track it
#    separates even more cleanly: 5.5-6.5 straight, 0.3-0.9 in corners.
#    All four are still computed and logged; only the returned one changes.
#
# 2. depth_law = 'ramp'
#    Linear interpolation from throttle_min at depth_min_m to throttle_max at
#    depth_max_m. Two tuned endpoints, no physics. This is the mechanism that
#    already existed behind use_depth_throttle, now reachable and fed a working
#    depth measure.
#
# 3. depth_law = 'sqrt'
#    v = sqrt(2 * a * d), the fastest speed from which the vehicle can still
#    stop inside the clearance it can see, then converted to a throttle command
#    through speed_per_throttle. Physically grounded rather than tuned, and the
#    right shape for a vehicle with no brake. Predicted from current logs:
#        dc 5.5 -> 6.7 m/s -> 0.275 throttle
#        dc 3.0 -> 4.8 m/s -> 0.198
#        dc 0.5 -> 1.3 m/s -> 0.054
#    Compare against the ramp at the same points: near-identical at 3.0 m,
#    markedly more cautious close in. Both are clipped to throttle_min and
#    throttle_max.
#
# 4. throttle_floor (default 0.05, applied only when a law is active)
#    At full lock steer_derate 0.7 leaves 30 per cent of a base the law may
#    already have cut to near zero. v8 showed what that produces: 0.011
#    throttle, nose against the wall at 0.26 m/s, no drive to rotate out.
#    The floor is applied after the derate so a corner can always be exited.
#
# HOW TO TEST
#   Baseline, must reproduce v9 exactly (8 laps, 0 collisions, 21.7 s):
#     ros2 run srm_racer gap_follower
#
#   Braking-distance law, conservative first pass:
#     ros2 run srm_racer gap_follower --ros-args \
#       -p depth_law:=sqrt -p throttle_max:=0.20
#   then raise throttle_max in steps: 0.20, 0.24, 0.28.
#
#   Linear ramp, for comparison on the same track:
#     ros2 run srm_racer gap_follower --ros-args \
#       -p depth_law:=ramp -p depth_min_m:=1.0 -p depth_max_m:=5.0 \
#       -p throttle_min:=0.10 -p throttle_max:=0.20
#
#   One parameter per run, 15+ laps, record. Watch the final turn: it already
#   passes as close as R=0.04 m and is the first thing that will fail.
#
# MEASURED ON THE PHASE 2 TRACK, 2026-09-15
# -----------------------------------------------------------------------------
#   depth_law off  (v9 defaults) .............. 21.7 s, 8 laps, 0 collisions
#   sqrt, tmax 0.16, margin 1.2, derate 0.85 .. 17.8 s, 8 laps, 0 collisions
#   sqrt, tmax 0.18, + bias_gain 0.20 ......... 17.5 s, clips the final turn
#   sqrt, tmax 0.20, + bias_gain 0.20 ......... 16.9 s, clips the final turn
#
#   BEST CLEAN CONFIGURATION IS THE 0.16 ROW. A collision costs +10 s, so a
#   clipped 16.9 scores 26.9 against a clean 17.8. Do not submit a clipping
#   configuration.
#
#   bias_gain 0.35 (tuned at 0.10 throttle) caused sustained weaving on the
#   straights at tmax 0.20: L and R swapped sides sample to sample and the
#   vehicle held ~15 per cent throttle instead of 20. The correction is a
#   steering angle and the displacement it produces scales with v^2, so the
#   gain does about four times the work at double the speed. 0.20 settled it.
#
#   depth_source da was tried for its lead time and is WORSE: throttle went
#   0.181 0.127 0.171 0.160 0.124 0.050 on consecutive samples. da is a wedge
#   about the AIM bearing, so it is a function of steering, and feeding it to
#   throttle closes the loop this node deliberately opened (see _throttle_for).
#   dc is steering-independent; keep it.
#
#   The final turn is entered off a straight, so dc stays saturated until the
#   wall is close. Widening centre_cone_deg from 4.0 is the steering-
#   independent way to buy lead time; at tmax 0.20 the law only needs about
#   4 m of dc, so capping it lower costs nothing.
#
# MEASURED AT 56 Hz - DISTRIBUTED MODE, 2026-09-16
# -----------------------------------------------------------------------------
# All earlier tuning was done at 15 Hz with the simulator and devkit on one
# machine. The organisers confirmed (Slack) that the native Linux simulator
# build causes this and that distributed computing mode is the fix: simulator
# on a second machine over LAN, devkit here. Measured 15.0 -> 56.3 Hz, median
# dt 66.7 -> 17.7 ms, max dt 103 -> 24 ms. Tune for 40-50 Hz per organisers.
#
#   v9 defaults .............................. 20.5 s clean  (21.7 at 15 Hz)
#   sqrt tmax 0.16 margin 1.2 derate 0.85 .... 16.4 s, 10 laps, 0 collisions
#   sqrt tmax 0.20 margin 1.2 derate 0.85 .... 15.5 s, 10 laps, 0 collisions
#
#   bias_gain is back at its 0.35 DEFAULT in both runs. The 0.20 value was a
#   workaround for straight-line weaving that does not exist at 56 Hz: logged
#   straights now hold L=0.77 R=0.77 with bias within +/-0.004 for consecutive
#   samples, while the term still contributes +0.19 to +0.26 through corners.
#   The weaving was stale feedback, not excessive gain.
#
#   throttle_max 0.20 clipped the final turn at 15 Hz and is clean at 56 Hz.
#   The grip limit did not change; command tracking did.
#
#   At dc 5.5 the sqrt law computes 0.243, so throttle_max below that is still
#   capping the straights. decel_margin_m is the next lever after the cap
#   stops binding, since it raises speed everywhere rather than on straights
#   alone.
#
# RESOLVED: open item 4, the 3.2 rad/s steering rate limit. Measured directly
# from the organisers' Phase 1 bag: p99 rate 3.335 rad/s, p99.9 6.356, max
# 12.435 - the limit is not enforced in simulation. Command-to-actual lag is
# 24 ms at r=0.999, tracking error p95 0.021 rad. Nothing to model.
#
# NOTE ON THE BAG: message rate there is 133 Hz on every topic, but LiDAR
# CONTENT updates at 39.8 Hz with 70.2 per cent duplicate payloads - the
# bridge restamps and republishes each scan. IMU and encoders are genuinely
# 133 Hz. Compare content rate, not message rate.
#
# NOT YET ADDRESSED
#   * speed_per_throttle 24.3 is a steady-state fit from two points. It says
#     nothing about how quickly the vehicle REACHES that speed, so the law
#     will command changes faster than the vehicle can follow. If the sqrt law
#     proves jumpy, a rate limit on throttle is the next thing to try.
#   * Corner-exit steering transient measured at 0.1 s: peaks decay
#     0.043 -> 0.009 -> 0.004 rad, i.e. damped, period about 2.3 s. Harmless at
#     0.10 throttle. Re-measure at higher speed, since the fixed 54 ms sensor
#     delay erodes phase margin as speed rises.
#
# REVISION NOTES (v9)
# -----------------------------------------------------------------------------
# NO ALGORITHMIC CHANGE. Six declared defaults are retuned for the Phase 2
# competition track (the bridge-silhouette layout in the compete simulator
# build), which has two near-90-degree corners and two sharp tower apexes,
# where Porto had neither. v8 defaults drove into the walls continuously on
# this layout: 18 collisions in 20 seconds.
#
# Each change was made and measured on its own. Result at the end: 8 laps,
# zero collisions, 21.7 s best lap.
#
#   fov_deg          60.0 -> 100.0
#     _prepare discards every sample outside the window before selection. The
#     openings at the two sharp corners sit near 80-90 degrees of bearing, so
#     at 60 they were deleted before the algorithm ever saw them and the
#     deepest remaining sample was the wall ahead. Logged steer was 0.00 rad
#     at 5.4 m/s entering a corner the LiDAR could see perfectly well. 130 was
#     also tried and gave no further gain, so 100 is the setting.
#
#   lookahead_m      1.90 -> 1.20
#     Pure pursuit saturates at alpha = 90 deg, so the largest angle the law
#     can EVER command is atan(2L / Ld). At Ld 1.90 that is 0.329 rad, only
#     63 per cent of the 0.5236 mechanical limit, bounding the tightest
#     achievable radius at L/tan(0.329) = 0.95 m. The apexes here are tighter
#     than that, so widening the field of view let the car see and aim at the
#     corner while the steering law still refused to turn hard enough. Full
#     authority needs Ld <= 2L/tan(0.5236) = 1.12 m; 1.20 gives 0.495 rad and
#     a 0.60 m minimum radius, which clears every corner on this track.
#
#   throttle         0.22 -> 0.10
#     At 5.4 m/s the tightest radius the tyres hold is v^2/a_lat ~ 2.98 m,
#     against apexes far tighter. The car was not mis-steering, it was
#     carrying speed it could not turn at. 0.14 was tried after the stack was
#     clean: 17.0 s best lap but wobbly with clipped corners, so 0.10 stands
#     until the depth ramp replaces the constant.
#
#   derate_exponent  1.0 -> 2.0
#     The argument for this was already written in _throttle_for and is
#     unchanged; v9 simply adopts it. Linear derate charges every small
#     correction full price: a commanded 0.22 was delivered as 0.179 at only
#     0.10 rad of steer. At exponent 2.0 a tenth of lock costs one per cent
#     instead of ten, and full lock still yields zero.
#
#   steer_derate     1.0 -> 0.7
#     Once lookahead_m unlocked near-full lock, the derate unlocked with it.
#     At steer_norm 0.944 the cut was 0.891, leaving 0.011 throttle: the car
#     reached corners it could now steer round and then STALLED in them,
#     nose against the wall at 0.26 m/s with no drive to rotate out. The
#     derate's justification is the lateral tyre curve peaking at 0.01 slip,
#     which is a grip argument that does not apply below 1 m/s. Capping the
#     cut at 70 per cent guarantees residual drive through a corner. 0.5 was
#     not needed; 0.7 solved it.
#
#   bias_gain        0.0 -> 0.35
#     _corridor_bias was written in v8 as diagnostic-only. It is now given
#     authority. The remaining collisions were an ENTRY LINE problem, not a
#     perception or steering one: the car arrived at both sharp corners
#     hugging the inside wall (logged L = 0.47 -> 0.40 -> 0.33 m while R rose
#     to 1.23), and _extend_disparities blanks atan(half_car / near_d) either
#     side of an edge, which at 0.33 m is 42 degrees. The corner was erased
#     by the extender because the car was too close to the wall going in.
#     Biasing toward the open side centres the car on the straight (logged
#     L = 0.77, R = 0.77 after the change) so the opening stays visible.
#     0.25 fixed the first corner but not the last; 0.35 fixed both.
#
# STILL OPEN AT THIS REVISION
#   * Straights run at 0.100 throttle with dc reading 5.5-6.5 m of clear road.
#     dc separates cleanly (median ~5.5 straight against 0.3-0.9 in corners),
#     so it is the right input for the ramp in _throttle_for. NOTE that
#     _path_depth currently returns d1, which measured +0.193 correlation
#     with speed (inverted, unusable); rewiring it to return dc is the
#     precondition for enabling use_depth_throttle.
#   * Bridge tick measured at 17.5 Hz on the compete build, max frame gap
#     0.103 s. A command held that long at 6 m/s is 62 cm of travel, which is
#     the most likely cause of run-to-run variation at corner entry.
#   * The final turn still passes close: logged R as low as 0.04 m. That is
#     the thinnest margin on the track and the first thing to fail when
#     speed rises.
#
# REVISION NOTES (v8)
# -----------------------------------------------------------------------------
# Two defects in v7 are addressed, and each is separately switchable so their
# effects can be measured independently.
#
# 1. LINEAR BEARING TO STEERING MAPPING.
#    v7 set the steering command to bearing / max_steer. A bearing of 50 deg
#    then demands 167 per cent of the mechanical limit and clips to full lock.
#    Logged output showed the command saturated at -1.000 through every corner.
#    Scaling the whole response down by a constant (steering_gain) suppressed
#    the saturation but also removed authority where a genuinely sharp
#    correction was needed, so the vehicle failed to recover after deep turns.
#
#    v8 restores the pure pursuit law
#        delta = atan(2 * L * sin(alpha) / Ld)
#    which compresses wide bearings without weakening the response to small
#    ones. It is the principled version of what a constant gain approximates.
#    Setting `use_pure_pursuit` false reverts to the v7 linear mapping.
#
# 2. ARGMAX PINNING AT THE FIELD OF VIEW BOUNDARY.
#    Logged aim bearings sat at exactly -0.873 rad with a 50 deg half-window,
#    and at exactly -1.571 rad with a 90 deg one, repeatedly. Those are the
#    window edges to the sample. Two causes combine: the deepest return
#    genuinely lies off to the side when the corridor turns, and np.diff cannot
#    detect a disparity at the first or last sample because that sample has no
#    neighbour inside the window, so a boundary reading is never extended.
#
#    v8 breaks ties centrally. Rather than taking the single deepest sample, it
#    collects every sample within `tie_tolerance_m` of the maximum and selects
#    the one nearest straight ahead. When several headings are effectively
#    equally open, the most forward of them is preferred, which stops the aim
#    drifting to a boundary that merely happens to read marginally longer.
#    Setting `tie_tolerance_m` to zero restores exact v7 argmax behaviour.
#
# ALGORITHM, once per incoming laser scan:
#   1. Restrict the scan to a forward field of view and clip ranges to a
#      horizon; treat non-returns as the horizon, since they denote open space.
#   2. Find disparities, meaning adjacent samples differing by more than a
#      threshold. Each marks an obstacle edge.
#   3. At each disparity, overwrite samples on the far side with the nearer
#      reading, across the angular span half a car width subtends at that
#      range. The LiDAR reasons as a single point but sits on a vehicle 0.27 m
#      wide, so a genuinely visible distant point can still put a wheel or
#      flank into the obstacle edge preceding it. Extending the disparity
#      removes every heading the vehicle cannot physically take.
#   4. Select the deepest remaining direction, preferring the most forward
#      among near-equal candidates.
#   5. Convert that bearing to a steering angle by pure pursuit.
#   6. Drive at a constant throttle.
#
# SPEED REMAINS CONSTANT IN THIS REVISION, deliberately. The purpose is still
# to establish that the steering geometry alone completes clean laps before any
# speed dynamics are reintroduced.
#
# DESIGN CONSTRAINTS
#   * Negative throttle engages REVERSE on this vehicle rather than braking, so
#     throttle is clamped non-negative at all times.
#   * Restricted topics (ips, odom, tf, lap and collision telemetry, and
#     reset_command) are never subscribed to, per section 2.4 of the rule book.
#   * The bridge tick observed on the development laptop is ~18.5 Hz against a
#     documented 40 Hz sensor rate, and the evaluation workstation will differ.
#     This revision holds no rate-dependent state, so it is unaffected.
#
# GEOMETRY (2026 Technical Guide)
#   Car width 0.270 m, wheelbase 0.324 m, front overhang 0.090 m.
#   LiDAR frame at x = 0.2733 m from the rear axle; front bumper at 0.414 m,
#   hence obstacles are 0.141 m closer to the bumper than the LiDAR reports.
#   Steering limits +/- 0.5236 rad. Scan is 1080 points over 270 degrees.
# =============================================================================

import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32

# --- Fixed vehicle geometry, do not tune -------------------------------------
LIDAR_TO_BUMPER_M = 0.141      # 0.414 m bumper - 0.2733 m LiDAR mount
CAR_HALF_WIDTH_M = 0.135       # 0.270 m overall width
WHEELBASE_M = 0.324
MAX_STEER_RAD = 0.5236         # +/- 30 deg mechanical limit


class GapFollower(Node):
    """Canonical disparity extender with pure pursuit steering."""

    def __init__(self):
        super().__init__('gap_follower')

        self.declare_parameter('fov_deg', 100.0)
        self.declare_parameter('horizon_m', 8.0)
        self.declare_parameter('disparity_threshold_m', 0.20)
        self.declare_parameter('extend_margin_m', 0.16)
        self.declare_parameter('tie_tolerance_m', 0.50)
        self.declare_parameter('use_pure_pursuit', True)
        self.declare_parameter('lookahead_m', 1.20)
        self.declare_parameter('steering_gain', 0.45)
        self.declare_parameter('throttle', 0.10)
        self.declare_parameter('use_depth_throttle', False)
        self.declare_parameter('throttle_min', 0.09)
        self.declare_parameter('throttle_max', 0.26)
        self.declare_parameter('depth_min_m', 2.0)
        self.declare_parameter('depth_max_m', 6.0)
        # v10: depth source and speed law
        self.declare_parameter('depth_source', 'dc')   # d1 | d2 | dc | da
        self.declare_parameter('depth_law', 'off')     # off | ramp | sqrt
        self.declare_parameter('speed_per_throttle', 24.3)
        self.declare_parameter('decel_mps2', 4.3)
        self.declare_parameter('decel_margin_m', 0.30)
        self.declare_parameter('throttle_floor', 0.05)
        self.declare_parameter('front_cone_deg', 15.0)
        self.declare_parameter('steer_derate', 0.7)
        self.declare_parameter('derate_exponent', 2.0)
        self.declare_parameter('path_half_width_m', 0.32)
        self.declare_parameter('arc_max_m', 8.0)
        self.declare_parameter('min_preview_radius_m', 3.0)
        self.declare_parameter('min_arc_m', 0.25)
        self.declare_parameter('centre_cone_deg', 4.0)
        self.declare_parameter('aim_cone_deg', 6.0)
        self.declare_parameter('log_period_s', 1.0)
        self.declare_parameter('bias_side_deg', 55.0)
        self.declare_parameter('bias_window_deg', 25.0)
        self.declare_parameter('bias_gain', 0.35)  # active since v9

        self._reload_parameters()
        self.last_log_s = 0.0
        self.depth_v1 = 0.0
        self.depth_v2 = 0.0
        self.depth_cone = 0.0
        self.depth_aim = 0.0

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.steer_pub = self.create_publisher(
            Float32, '/autodrive/roboracer_1/steering_command', 10)
        self.throttle_pub = self.create_publisher(
            Float32, '/autodrive/roboracer_1/throttle_command', 10)
        self.scan_sub = self.create_subscription(
            LaserScan, '/autodrive/roboracer_1/lidar', self.scan_callback, qos)

        self.get_logger().info('gap_follower v10 ready, waiting for laser scans')

    def _reload_parameters(self):
        """Pull current parameter values into plain attributes each cycle."""
        g = self.get_parameter
        self.fov_rad = math.radians(g('fov_deg').value)
        self.horizon_m = g('horizon_m').value
        self.disparity_threshold_m = g('disparity_threshold_m').value
        self.extend_margin_m = g('extend_margin_m').value
        self.tie_tolerance_m = g('tie_tolerance_m').value
        self.use_pure_pursuit = g('use_pure_pursuit').value
        self.lookahead_m = g('lookahead_m').value
        self.steering_gain = g('steering_gain').value
        self.throttle = g('throttle').value
        self.use_depth_throttle = g('use_depth_throttle').value
        self.throttle_min = g('throttle_min').value
        self.throttle_max = g('throttle_max').value
        self.depth_min_m = g('depth_min_m').value
        self.depth_max_m = g('depth_max_m').value
        self.depth_source = str(g('depth_source').value).lower()
        self.depth_law = str(g('depth_law').value).lower()
        self.speed_per_throttle = g('speed_per_throttle').value
        self.decel_mps2 = g('decel_mps2').value
        self.decel_margin_m = g('decel_margin_m').value
        self.throttle_floor = g('throttle_floor').value
        self.front_cone_rad = math.radians(g('front_cone_deg').value)
        self.steer_derate = g('steer_derate').value
        self.derate_exponent = g('derate_exponent').value
        self.path_half_width_m = g('path_half_width_m').value
        self.arc_max_m = g('arc_max_m').value
        self.min_preview_radius_m = g('min_preview_radius_m').value
        self.min_arc_m = g('min_arc_m').value
        self.centre_cone_rad = math.radians(g('centre_cone_deg').value)
        self.aim_cone_rad = math.radians(g('aim_cone_deg').value)
        self.log_period_s = g('log_period_s').value
        self.bias_side_rad = math.radians(g('bias_side_deg').value)
        self.bias_window_rad = math.radians(g('bias_window_deg').value)
        self.bias_gain = g('bias_gain').value

    def scan_callback(self, msg: LaserScan):
        """Main control loop, executed once per laser scan."""
        self._reload_parameters()

        ranges, angles = self._prepare(msg)
        if ranges.size == 0:
            self._publish(0.0, 0.0)
            return

        self._extend_disparities(ranges, msg.angle_increment)

        bias_rad, free_l, free_r = self._corridor_bias(ranges, angles)

        best = self._select_heading(ranges, angles)
        target_rad = float(angles[best]) + bias_rad
        steer_norm = self._steering_command(target_rad)

        depth_m = self._path_depth(ranges, angles, steer_norm)
        throttle = self._throttle_for(depth_m, steer_norm)

        self._log_state(msg, target_rad, depth_m, steer_norm, throttle,
                        bias_rad, free_l, free_r)
        self._publish(steer_norm, throttle)

    def _prepare(self, msg):
        """
        Return bumper-referenced ranges and their bearings across the forward
        field of view. Non-returns arrive as +inf and are treated as the
        horizon, since open space is exactly what they represent.
        """
        r = np.asarray(msg.ranges, dtype=np.float64)
        r = np.nan_to_num(r, nan=0.0, posinf=self.horizon_m, neginf=0.0)
        r = np.maximum(r - LIDAR_TO_BUMPER_M, 0.0)
        r = np.minimum(r, self.horizon_m)

        angles = msg.angle_min + np.arange(r.size) * msg.angle_increment
        keep = np.abs(angles) <= self.fov_rad
        return r[keep], angles[keep]

    def _extend_disparities(self, ranges, angle_increment):
        """
        Overwrite samples adjacent to each obstacle edge with the nearer
        reading, across the angular span half a car width subtends at that
        range.

        For a disparity between samples i and i+1, the nearer of the two marks
        an edge the vehicle must clear. The number of samples to overwrite is

            n = arctan(half_width / d) / angle_increment

        where d is the nearer range. Overwriting proceeds away from the nearer
        sample. Using np.minimum rather than assignment means overlapping
        extensions from several disparities compose correctly instead of the
        last one written winning.
        """
        half_car = CAR_HALF_WIDTH_M + self.extend_margin_m
        diffs = np.diff(ranges)

        for i in np.flatnonzero(np.abs(diffs) > self.disparity_threshold_m):
            if diffs[i] > 0:
                near_idx, near_d, step = i, ranges[i], 1
            else:
                near_idx, near_d, step = i + 1, ranges[i + 1], -1

            if near_d < 1e-3:
                continue

            n = int(math.atan2(half_car, near_d) / angle_increment) + 1

            if step > 0:
                lo, hi = near_idx, min(near_idx + n + 1, ranges.size)
            else:
                lo, hi = max(near_idx - n, 0), near_idx + 1

            ranges[lo:hi] = np.minimum(ranges[lo:hi], near_d)

    def _select_heading(self, ranges, angles):
        """
        Return the index of the chosen heading.

        Taking a plain argmax lets the aim settle on whichever sample happens
        to read marginally longest, which in practice is often at the edge of
        the window: the corridor genuinely opens sideways in a corner, and a
        boundary sample can never be disparity-extended because np.diff has no
        neighbour for it. Collecting all samples within a tolerance of the
        maximum and choosing the most forward of them keeps the aim ahead
        whenever several headings are effectively equally open.
        """
        if self.tie_tolerance_m <= 0.0:
            return int(np.argmax(ranges))

        peak = float(np.max(ranges))
        near_max = np.flatnonzero(ranges >= peak - self.tie_tolerance_m)
        return int(near_max[np.argmin(np.abs(angles[near_max]))])

    def _steering_command(self, target_rad):
        """
        Convert an aim bearing to a normalised steering command.

        Pure pursuit gives delta = atan(2 * L * sin(alpha) / Ld), which cannot
        demand more lock than the geometry justifies. A 60 deg bearing at a
        1.5 m lookahead yields 0.35 rad rather than the 1.05 rad a linear
        mapping would ask for, while small bearings remain close to linear.
        """
        if self.use_pure_pursuit:
            delta = math.atan2(
                2.0 * WHEELBASE_M * math.sin(target_rad), self.lookahead_m)
        else:
            delta = self.steering_gain * target_rad

        return float(np.clip(delta / MAX_STEER_RAD, -1.0, 1.0))

    def _front_depth(self, ranges, angles):
        """
        Minimum free distance inside a narrow cone straight ahead.

        Depth at the aim index is the wrong input for a speed decision. Part
        way through a corner the aim points at open track 40 to 58 degrees off
        axis, so that reading saturates near the sensing horizon while the
        vehicle is still turning, and the throttle ramp responds as though the
        road were straight. Logged telemetry showed 0.21 throttle at an aim
        bearing of 58 degrees, which is power on mid-corner and produced
        repeated exit collisions.

        Taking the minimum over a narrow forward cone answers the question the
        speed controller actually needs: how far can this vehicle travel along
        its current heading before the corridor constrains it.
        """
        mask = np.abs(angles) <= self.front_cone_rad
        if not mask.any():
            return float(np.min(ranges))
        return float(np.min(ranges[mask]))

    def _corridor_bias(self, ranges, angles):
        """
        Measure the lateral asymmetry of the corridor and derive a steering
        bias from it. DIAGNOSTIC ONLY at bias_gain 0.0.

        A gap follower that aims at the deepest visible point traces a path
        near the centre of the corridor. That is not a racing line: a racing
        line runs wide on entry, tightens to an apex, and opens out again on
        exit, so that the vehicle is close to straight for much of the corner.
        Logged telemetry shows the present aim bearing sitting between 0.4 and
        0.9 rad almost continuously, meaning the vehicle is steering, and
        therefore derating throttle, essentially all the time.

        The corner geometry cannot be known ahead of time from an 8 m scan, but
        its asymmetry can be measured. Free distance is averaged over a window
        centred on +bias_side_deg (left) and again on -bias_side_deg (right).
        Approaching a right hand corner the left side is the more open of the
        two, at the apex the two are comparable and both short, and on exit the
        right opens up again. The normalised difference

            bias = (free_left - free_right) / (free_left + free_right)

        is therefore positive when the open space lies to the left, negative
        when it lies to the right, and near zero on a straight or at an apex.
        Adding a multiple of it to the aim bearing pushes the vehicle toward
        the outside of the corner on entry and lets it run out again on exit.

        Returns the bias in radians together with the two raw averages, so the
        measure can be validated against observed behaviour before it is given
        any authority over steering.
        """
        half = self.bias_window_rad

        left = np.abs(angles - self.bias_side_rad) <= half
        right = np.abs(angles + self.bias_side_rad) <= half

        free_l = float(np.mean(ranges[left])) if left.any() else 0.0
        free_r = float(np.mean(ranges[right])) if right.any() else 0.0

        total = free_l + free_r
        if total < 1e-3:
            return 0.0, free_l, free_r

        normalised = (free_l - free_r) / total
        return self.bias_gain * normalised, free_l, free_r

    def _path_depth_v1(self, ranges, angles, steer_norm):
        """
        Free distance measured along the arc the vehicle is actually following.

        Two earlier measures were tried and both are wrong in opposite ways.

        Taking the range at the aim index is too optimistic: part way through a
        corner the aim points at open track 40 to 58 degrees off axis, so the
        reading saturates near the sensing horizon while the vehicle is still
        turning. Logged telemetry showed 0.21 throttle at an aim bearing of
        58 degrees, which is power on mid-corner, and produced repeated exit
        collisions.

        Taking the minimum over a cone fixed to the chassis is too pessimistic:
        a cone locked to the current heading points straight at the outside
        barrier through a corner, so depth collapsed below 0.5 m for much of
        the lap and throttle pinned at its floor. Lap time regressed from 8.8 s
        to 12.0 s.

        The correct question is how far this vehicle can travel before the
        corridor constrains it, and that depends on the path, not the heading.
        The steering angle implies a turn radius R = L / tan(delta), so the
        vehicle is following a circle of that radius centred abeam. A scan
        point at range r and bearing theta lies at (r cos theta, r sin theta);
        its distance from the arc centre determines whether the vehicle will
        pass through it. Points whose radial offset from the arc exceeds half
        the swept path width are missed and are ignored. Of the remainder, the
        nearest measured along the arc is the true constraint.

        As delta tends to zero the radius diverges and the test degenerates to
        a straight corridor of the same width, which is the correct limiting
        behaviour.
        """
        delta = steer_norm * MAX_STEER_RAD
        tan_delta = math.tan(delta)

        xs = ranges * np.cos(angles)
        ys = ranges * np.sin(angles)

        if abs(tan_delta) < 1e-3:
            # Straight ahead: a rectangular corridor of the swept width.
            on_path = (np.abs(ys) <= self.path_half_width_m) & (xs > 0.0)
            if not on_path.any():
                return self.arc_max_m
            return float(np.clip(np.min(xs[on_path]), 0.0, self.arc_max_m))

        radius = WHEELBASE_M / tan_delta          # signed; left turn positive
        centre_y = radius

        # Radial distance of each point from the arc centre.
        offset = np.abs(np.hypot(xs, ys - centre_y) - abs(radius))
        on_path = offset <= self.path_half_width_m

        # Angle subtended from the centre, measured forward along the arc.
        phi = np.arctan2(xs, np.sign(radius) * (centre_y - ys))
        forward = phi > 0.0

        valid = on_path & forward
        if not valid.any():
            return self.arc_max_m

        arc_len = abs(radius) * phi[valid]
        return float(np.clip(np.min(arc_len), 0.0, self.arc_max_m))

    def _path_depth_v2(self, ranges, angles, steer_norm):
        """
        Corrected free distance along the arc the vehicle is following.

        Differs from _path_depth in three ways; see the revision notes for the
        evidence behind each.

        First, the bumper offset is treated as a translation rather than a
        radial shrink. _prepare returns bumper-referenced ranges, so the raw
        range is recovered before converting to Cartesian and the offset is
        then applied along x only. Points with x <= 0 lie behind the bumper
        plane and are discarded rather than counted as obstacles ahead.

        Second, the projection radius is floored. The instantaneous radius
        R = L / tan(delta) reaches 1.16 m at the lock this vehicle actually
        uses, and a circle that tight fits inside the corridor without
        touching it, so no constraint is found and the measure saturates. Over
        a preview of several metres the steering will change substantially, so
        a floored radius is the better model of average curvature.

        Third, constraints closer than min_arc_m are ignored, so the vehicle's
        own wheels and bodywork cannot pin the measure at zero.
        """
        # --- true Cartesian relative to the front bumper --------------------
        r_raw = ranges + LIDAR_TO_BUMPER_M
        xs = r_raw * np.cos(angles) - LIDAR_TO_BUMPER_M
        ys = r_raw * np.sin(angles)

        ahead = xs > 0.0
        if not ahead.any():
            return self.arc_max_m

        delta = steer_norm * MAX_STEER_RAD
        tan_delta = math.tan(delta)

        if abs(tan_delta) < 1e-3:
            on_path = ahead & (np.abs(ys) <= self.path_half_width_m)
            if not on_path.any():
                return self.arc_max_m
            d = xs[on_path]
            d = d[d >= self.min_arc_m]
            if d.size == 0:
                return self.arc_max_m
            return float(np.clip(np.min(d), 0.0, self.arc_max_m))

        radius = WHEELBASE_M / tan_delta          # signed; left turn positive
        if abs(radius) < self.min_preview_radius_m:
            radius = math.copysign(self.min_preview_radius_m, radius)
        centre_y = radius

        offset = np.abs(np.hypot(xs, ys - centre_y) - abs(radius))
        on_path = ahead & (offset <= self.path_half_width_m)

        phi = np.arctan2(xs, np.sign(radius) * (centre_y - ys))
        valid = on_path & (phi > 0.0)
        if not valid.any():
            return self.arc_max_m

        arc_len = abs(radius) * phi[valid]
        arc_len = arc_len[arc_len >= self.min_arc_m]
        if arc_len.size == 0:
            return self.arc_max_m
        return float(np.clip(np.min(arc_len), 0.0, self.arc_max_m))

    def _aim_from_steer(self, steer_norm):
        """
        Recover the aim bearing from the steering command.

        Pure pursuit gives delta = atan(2 L sin(alpha) / Ld), so
        sin(alpha) = tan(delta) Ld / (2 L). Inverting here avoids changing
        the scan_callback signature. Verified against logged telemetry: a
        command of 0.519 recovers 1.036 rad against a logged aim of 1.034.
        """
        if not self.use_pure_pursuit:
            return steer_norm * MAX_STEER_RAD / max(self.steering_gain, 1e-6)
        delta = steer_norm * MAX_STEER_RAD
        s = math.tan(delta) * self.lookahead_m / (2.0 * WHEELBASE_M)
        return float(math.asin(float(np.clip(s, -1.0, 1.0))))

    def _depth_cone(self, ranges, angles, centre_rad, half_width_rad):
        """
        Minimum bumper-referenced range inside a wedge about a bearing.

        Used for two of the candidate measures: a narrow wedge straight ahead,
        which is the quantity the simulator HUD reports as a single LiDAR
        measurement, and a wedge about the aim bearing.
        """
        mask = np.abs(angles - centre_rad) <= half_width_rad
        if not mask.any():
            return self.arc_max_m
        return float(np.clip(np.min(ranges[mask]), 0.0, self.arc_max_m))

    def _path_depth(self, ranges, angles, steer_norm):
        """
        Comparison wrapper. Computes four candidate depth measures, stores
        three of them for logging, and returns the ORIGINAL so the control
        path is unchanged while all four are compared on the same runs.

          d1  original arc projection
          d2  corrected arc projection
          dc  narrow wedge straight ahead
          da  wedge about the aim bearing

        A usable measure must fall as the corridor closes AND lead the
        steering, so that throttle is already dropping while the corner is
        still several metres away. Deceleration is by idle torque alone.
        """
        self.depth_v1 = self._path_depth_v1(ranges, angles, steer_norm)
        self.depth_v2 = self._path_depth_v2(ranges, angles, steer_norm)
        self.depth_cone = self._depth_cone(
            ranges, angles, 0.0, self.centre_cone_rad)
        self.depth_aim = self._depth_cone(
            ranges, angles, self._aim_from_steer(steer_norm),
            self.aim_cone_rad)
        return {
            'd1': self.depth_v1,
            'd2': self.depth_v2,
            'dc': self.depth_cone,
            'da': self.depth_aim,
        }.get(self.depth_source, self.depth_cone)

    def _throttle_for(self, depth_m, steer_norm):
        """
        Scale throttle with the free distance at the chosen heading.

        After disparity extension the depth at the aim index measures how far
        the vehicle can travel before the corridor constrains it: on a straight
        it saturates at the sensing horizon, and entering a corner it
        collapses. Ramping throttle linearly between two depth thresholds
        therefore brakes on corner approach and accelerates on exit, using
        nothing but the current scan.

        This is deliberately open loop. An earlier revision derived a speed
        target from the steering angle and closed a loop on measured speed,
        which coupled speed to lookahead to aim bearing to curvature and back
        to speed; logged throttle showed a self-sustaining oscillation of about
        0.8 s that no amount of filtering or rate limiting removed. Here the
        command depends only on the scan, so there is no path from the
        vehicle's own motion back into its own command and the loop cannot
        close.

        Negative throttle is never emitted because it engages reverse rather
        than braking on this vehicle. Deceleration is by idle torque alone,
        which is why the ramp must begin well before the corner: depth_min_m
        and depth_max_m should be set so that throttle is already falling
        while the corner is still several metres away.
        """
        if self.depth_law == 'ramp':
            span = max(self.depth_max_m - self.depth_min_m, 1e-3)
            frac = (depth_m - self.depth_min_m) / span
            frac = float(np.clip(frac, 0.0, 1.0))
            throttle = self.throttle_min + frac * (
                self.throttle_max - self.throttle_min)

        elif self.depth_law == 'sqrt':
            # Fastest speed from which the vehicle can still stop within the
            # clearance it can currently see:  v = sqrt(2 * a * d).
            #
            # a is the MEASURED idle coastdown, 4.3 m/s^2 (coastdown.py); there
            # is no brake on this vehicle and negative throttle engages reverse,
            # so idle torque is the only deceleration available and this is the
            # correct conservative form. decel_margin_m keeps a stopping buffer
            # short of the obstacle itself.
            #
            # speed_per_throttle converts the speed target into a command. The
            # constant is measured, not assumed: logged steady state gave
            # 0.10 -> 2.49 m/s and 0.22 -> 5.40 m/s, a slope of 24.25 m/s per
            # unit throttle with a negligible intercept. Re-measure it if the
            # vehicle model changes.
            usable = max(depth_m - self.decel_margin_m, 0.0)
            v_target = math.sqrt(2.0 * self.decel_mps2 * usable)
            throttle = v_target / max(self.speed_per_throttle, 1e-3)
            throttle = float(np.clip(
                throttle, self.throttle_min, self.throttle_max))

        else:
            throttle = self.throttle

        # Cut power with steering angle, raised to an exponent.
        #
        # A linear derate penalises every correction equally, so the small
        # inputs used to hold a straight line cost real speed: logged telemetry
        # showed a commanded 0.22 delivered as 0.179 at only 0.10 rad of steer.
        # Raising the normalised steering to a power leaves those corrections
        # nearly untouched while preserving the full cut at lock. At exponent
        # 2.0 a tenth of lock costs one per cent instead of ten, and full lock
        # still yields zero throttle.
        #
        # The cut is needed at all because the lateral tire curve peaks at 0.01
        # slip and has halved by 0.10, so applying drive torque while cornering
        # hard pushes the vehicle past the grip peak and it runs wide on exit.
        cut = self.steer_derate * (abs(steer_norm) ** self.derate_exponent)
        throttle *= max(1.0 - cut, 0.0)

        # Floor. At full lock with steer_derate 0.7 the derate leaves 30 per
        # cent of a base that the law may itself have cut to near zero, and the
        # vehicle then STALLS in the corner: nose against the wall, no drive to
        # rotate out. Observed directly at v8 defaults. The floor only applies
        # where a law is active, so v9 behaviour is unchanged.
        if self.depth_law != 'off':
            throttle = max(throttle, self.throttle_floor)
        return max(throttle, 0.0)

    def _log_state(self, msg, target_rad, depth_m, steer_norm, throttle,
                   bias_rad=0.0, free_l=0.0, free_r=0.0):
        """Emit a one-line state summary about once per second."""
        now_s = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if now_s - self.last_log_s < self.log_period_s:
            return
        self.last_log_s = now_s
        self.get_logger().info(
            'aim={:+.3f}  steer={:+.3f}  thr={:.3f}  '
            'd1={:.2f} d2={:.2f} dc={:.2f} da={:.2f}  '
            'L={:.2f} R={:.2f} bias={:+.3f}'.format(
                target_rad, steer_norm, throttle,
                self.depth_v1, self.depth_v2, self.depth_cone,
                self.depth_aim,
                free_l, free_r, bias_rad))

    def _publish(self, steer_norm, throttle):
        steer_msg = Float32()
        steer_msg.data = float(np.clip(steer_norm, -1.0, 1.0))
        self.steer_pub.publish(steer_msg)

        throttle_msg = Float32()
        throttle_msg.data = float(np.clip(throttle, 0.0, 1.0))
        self.throttle_pub.publish(throttle_msg)


def main(args=None):
    rclpy.init(args=args)
    node = GapFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
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