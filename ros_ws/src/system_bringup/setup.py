import os
from glob import glob

from setuptools import find_packages, setup

package_name = "system_bringup"

setup(
    name=package_name,
    version="1.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            os.path.join("share", package_name, "launch"),
            [f for f in glob("launch/*") if os.path.isfile(f)],
        ),
        (
            os.path.join("lib", package_name),
            [f for f in glob("scripts/*") if os.path.isfile(f)],
        ),
        *[
            (os.path.join("share", package_name, os.path.dirname(f)), [f])
            for f in glob("config/**", recursive=True)
            if os.path.isfile(f)
        ],
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="IMPACT Authors",
    maintainer_email="winstongu20@gmail.com",
    description="Minimal bringup flows for system-level controller profiling.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "set_collision_behavior = system_bringup.set_collision_behavior:main",
            "franka_gripper_bridge = system_bringup.franka_gripper_bridge:main",
            "dataset_recorder = system_bringup.dataset_recorder:main",
            "pose_error_broadcaster = system_bringup.pose_error_broadcaster:main",
        ]
    },
)
