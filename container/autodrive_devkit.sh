#!/bin/bash
set -e
# =============================================================================
# autodrive_devkit.sh - CEM Navigators competition entrypoint
# =============================================================================
# Starts the devkit bridge, then the localisation and tracking stack.
#
#   particle_filter   LiDAR + encoders + IMU against the prebuilt map.
#                     No ground truth: /ips, /odom and /tf are not read.
#   raceline_tracker  pure pursuit on the stored line, speed from the profile.
#
# The lookahead is capped at 1.1 m, below the corridor width. At the default
# 3.0 m the aim point can cross the thin barrier between the deck and the
# upper span and the vehicle steers into it.
# =============================================================================

source /opt/ros/humble/setup.bash
source /home/autodrive_devkit/install/setup.bash
cd /home/autodrive_devkit

MAPS=/home/autodrive_devkit/install/srm_racer/share/srm_racer/maps

ros2 launch autodrive_roboracer bringup_headless.launch.py &
sleep 3

ros2 run srm_racer particle_filter "$MAPS/bridge_ips_cl.yaml" -b 120 -n 3000 &
sleep 5

ros2 run srm_racer raceline_tracker "$MAPS/raceline_v17_r.csv"
