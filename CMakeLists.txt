# Copyright (c) 2023, Horizon Robotics.
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

cmake_minimum_required(VERSION 3.5)
project(hobot_llm)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)

include_directories(
  include
  ${CMAKE_SYSROOT}/usr/include/python3.8
)

link_directories(
  ${CMAKE_SYSROOT}/usr/lib/hbbpu/
)

add_executable(hobot_llm src/main.cpp src/hobot_llm.cpp src/model/bpu_bloom_model.cpp src/tokenizer/tokenization_bloom.cpp)
ament_target_dependencies(hobot_llm rclcpp std_msgs)
target_link_libraries(hobot_llm easy_dnn dnn hlog -lpython3.8)

add_executable(hobot_llm_chat src/hobot_llm_chat.cpp src/model/bpu_bloom_model.cpp src/tokenizer/tokenization_bloom.cpp)
target_link_libraries(hobot_llm_chat easy_dnn dnn hlog -lpython3.8)

install(TARGETS
  hobot_llm
  DESTINATION lib/${PROJECT_NAME})

install(TARGETS
  hobot_llm_chat
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY
  ${CMAKE_CURRENT_SOURCE_DIR}/src/tokenizer/tokenization_bloom_py
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
