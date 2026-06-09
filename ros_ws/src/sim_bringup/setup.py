import os
from glob import glob

from setuptools import find_packages, setup

package_name = "sim_bringup"

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
        *[
            (os.path.join("share", package_name, os.path.dirname(f)), [f])
            for f in glob("assets/**", recursive=True)
            if os.path.isfile(f)
        ],
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
    description="Simulation bringup flows for MuJoCo data collection.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "pick_place_demo = sim_bringup.pick_place_demo:main",
            "video_recorder = sim_bringup.video_recorder:main",
            "dataset_recorder = sim_bringup.dataset_recorder:main",
            "fake_franka_gripper = sim_bringup.fake_franka_gripper:main",
            "policy_deployment = sim_bringup.policy_deployment:main",
        ]
    },
)
