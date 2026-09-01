#!/usr/bin/env python3
# =============================================================================
# File        : racing.launch.py
# Package     : srm_racer
# Project     : RoboRacer Sim Racing League @ IROS 2026 (AutoDRIVE Ecosystem)
# Target      : ROS 2 Humble / Python 3.10
# Created     : 2026-08-31
# =============================================================================
# CODE DESCRIPTION
# -----------------------------------------------------------------------------
# Launches the srm_racer autonomous racing stack. The AutoDRIVE devkit bridge
# is launched separately (bringup_headless.launch.py or bringup_graphics.
# launch.py) and is not started here, so that this file can be invoked from the
# competition entrypoint script alongside the organiser-provided bringup.
# =============================================================================

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='srm_racer',
            executable='gap_follower',
            name='gap_follower',
            output='screen',
            parameters=[{
                'throttle_cruise': 0.06,
                'bubble_radius_m': 0.30,
                'gap_threshold_m': 1.0,
            }],
        ),
    ])
