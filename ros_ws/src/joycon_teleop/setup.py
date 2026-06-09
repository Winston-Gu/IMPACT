import os
from glob import glob

from setuptools import find_packages, setup

package_name = "joycon_teleop"

setup(
    name=package_name,
    version="1.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        # Include launch files
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        # Include RViz config
        (os.path.join("share", package_name, "rviz"), glob("rviz/*.rviz")),
    ],
    install_requires=["setuptools"],
    zip_safe=False,
    maintainer="IMPACT Authors",
    maintainer_email="winstongu20@gmail.com",
    description="Joy-Con teleoperation for robot control",
    license="MIT",
    entry_points={
        "console_scripts": [
            "joycon_teleop_node = joycon_teleop.joycon_teleop_node:main",
            "joycon_single_teleop_node = joycon_teleop.joycon_single_teleop_node:main",
            "joycon_dual_teleop_node = joycon_teleop.joycon_dual_teleop_node:main",
            "trajectory_recorder = joycon_teleop.trajectory_recorder:main",
            "joycon_check = joycon_teleop.joycon_check:main",
        ],
    },
)
