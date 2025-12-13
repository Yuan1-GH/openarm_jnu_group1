from setuptools import setup
import os
from glob import glob

package_name = 'arm'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

        # 安装 assets 里面的所有文件
        ('share/' + package_name + '/assets/011_banana',
            glob('arm/assets/011_banana/*')),

        # 安装 scene
        ('share/' + package_name + '/scene',
            glob('arm/scene/*')),

        # 安装 config
        ('share/' + package_name + '/config',
            glob('arm/config/*')),

        # 安装 launch
        ('share/' + package_name + '/launch',
            glob('arm/launch/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yuan',
    maintainer_email='xxx@example.com',
    description='arm simulation package',
    license='MIT',
    entry_points={
        'console_scripts': [
            'sim_node = arm.scripts.sim_node:main',
        ],
    },
)
