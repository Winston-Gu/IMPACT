cmake_minimum_required(VERSION 3.5)
project(predictive_controller)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Default to release build
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(ament_cmake REQUIRED)
find_package(controller_interface REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(generate_parameter_library REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(pinocchio REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(realtime_tools REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)

generate_parameter_library(
  cartesian_impedance_controller_parameters
  src/cartesian_impedance_controller.yaml
)

generate_parameter_library(
  pose_broadcaster_parameters
  src/pose_broadcaster.yaml
)

generate_parameter_library(
  residual_observer_controller_parameters
  src/residual_observer_controller.yaml
)

generate_parameter_library(
  cerebellum_predictive_controller_parameters
  src/cerebellum_predictive_controller.yaml
)

add_library(
  ${PROJECT_NAME}
  SHARED
  src/cartesian_impedance_controller.cpp
  src/noop_effort_controller.cpp
  src/pose_broadcaster.cpp
  src/residual_observer_controller.cpp
  src/cerebellum_predictive_controller.cpp
)

target_include_directories(
  ${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(
  ${PROJECT_NAME}
  controller_interface
  generate_parameter_library
  geometry_msgs
  pinocchio
  pluginlib
  rclcpp
  rclcpp_lifecycle
  realtime_tools
  sensor_msgs
  std_msgs
)

target_link_libraries(
  ${PROJECT_NAME}
  Eigen3::Eigen
  cartesian_impedance_controller_parameters
  pose_broadcaster_parameters
  residual_observer_controller_parameters
  cerebellum_predictive_controller_parameters
)

pluginlib_export_plugin_description_file(
  controller_interface predictive_controller.xml)

install(
  TARGETS
    ${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(
  DIRECTORY include/
  DESTINATION include
)

ament_export_include_directories(
  include
)
ament_export_libraries(
  ${PROJECT_NAME}
)
ament_export_dependencies(
  controller_interface
  Eigen3
  generate_parameter_library
  geometry_msgs
  pinocchio
  pluginlib
  rclcpp
  rclcpp_lifecycle
  realtime_tools
  sensor_msgs
  std_msgs
)

ament_package()
