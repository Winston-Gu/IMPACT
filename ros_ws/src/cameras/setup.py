from setuptools import setup

package_name = "cameras"

setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="IMPACT Authors",
    maintainer_email="winstongu20@gmail.com",
    description="Nodes for running cameras.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "realsense = cameras.realsense:main",
        ],
    },
)
