# =============================================================================
# File        : setup.py
# Package     : srm_racer
# Project     : RoboRacer Sim Racing League @ IROS 2026 (AutoDRIVE Ecosystem)
# Target      : ROS 2 Humble / Python 3.10
# Created     : 2026-08-31
# =============================================================================
# CODE DESCRIPTION
# -----------------------------------------------------------------------------
# ament_python build configuration for the srm_racer package. Registers the
# gap_follower node as a console entry point and installs the launch files.
# This package is a sibling of the organiser-provided autodrive_devkit package
# and does not modify it in any way, as required by the competition rules.
# =============================================================================

import os
from glob import glob
from setuptools import setup

package_name = 'srm_racer'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Shyam',
    maintainer_email='shyam@example.com',
    description='Autonomous racing stack for RoboRacer Sim Racing League, IROS 2026.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'gap_follower = srm_racer.gap_follower:main',
        ],
    },
)
