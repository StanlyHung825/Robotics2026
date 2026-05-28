import os
from glob import glob

from setuptools import setup

package_name = "myrobot"

setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "urdf"), glob("urdf/*")),
        (os.path.join("share", package_name, "meshes"), glob("meshes/*")),
        (os.path.join("share", package_name, "config"), glob("config/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="ldsc",
    maintainer_email="nthuldsc@gmail.com",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "kb_control = myrobot.kb_control:main",
            # "moveit_real_arm_interface = myrobot.0_moveit_real_arm_interface:main",
            "serial_with_st = myrobot.0_serial_with_ST:main",
            "moveit_real_arm_interface = myrobot.0_moveit_real_arm_interface:main",
            "magnet_moveit_real_arm_interface = myrobot.0_magnet_moveit_real_arm_interface:main",
            "IK_path_planning = myrobot.0_IK_path_planning:main",
            "hanoi_planner = myrobot.0_hanoi_planner:main",
            "hanoi_spawn_objects = myrobot.0_hanoi_spawn_objects:main",
        ],
    },
)
