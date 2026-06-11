import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'myrobot'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*')),
        (os.path.join('share', package_name, 'meshes'), glob('meshes/*')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jail487',
    maintainer_email='jackson4771@gapp.nthu.edu.tw',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'hanoi_vision = myrobot.hanoi_vision_status_node:main',
            'hanoi_status_listener = myrobot.hanoi_status_listener:main',
            'hanoi_status_to_sim = myrobot.hanoi_spawn_from_status:main',
            'hanoi_planner = myrobot.0_hanoi_planner:main',
            'hanoi_spawn_objects = myrobot.0_hanoi_spawn_objects:main',
            'magnet_moveit_real_arm_interface = myrobot.0_magnet_moveit_real_arm_interface:main',
            'magnet_serial_with_ST = myrobot.0_magnet_serial_with_ST:main',
            'hanoi_coordinator = myrobot.hanoi_coordinator:main',
        ],
    },
)
