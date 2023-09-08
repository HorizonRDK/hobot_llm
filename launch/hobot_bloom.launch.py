# Copyright (c) 2023，Horizon Robotics.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'hobot_llm_topic_sub',
            default_value='/text_query',
            description='hobot llm subscribe topic name'),
        DeclareLaunchArgument(
            'hobot_llm_topic_pub',
            default_value='/text_result',
            description='hobot llm publish topic name'),
        Node(
            package='hobot_llm',
            executable='hobot_llm',
            output='screen',
            parameters=[
                {"topic_sub": LaunchConfiguration('hobot_llm_topic_sub')},
                {"topic_pub": LaunchConfiguration('hobot_llm_topic_pub')}
            ],
            arguments=['--ros-args', '--log-level', 'error']
        )
    ])
