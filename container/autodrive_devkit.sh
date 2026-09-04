#!/bin/bash
set -e
# Setup Development Environment
source /opt/ros/humble/setup.bash
source /home/autodrive_devkit/install/setup.bash
# AutoDRIVE Devkit Workspace
cd /home/autodrive_devkit
# Launch AutoDRIVE Devkit Headless
ros2 launch autodrive_roboracer bringup_headless.launch.py &
# Launch CEM Navigators racing stack
sleep 3
ros2 run srm_racer gap_follower
