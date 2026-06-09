// Copyright (c) 2025
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <algorithm>
#include <string>
#include <unordered_set>

#include <Eigen/Dense>
#include <controller_interface/controller_interface.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>
#include <rclcpp/publisher.hpp>
#include <rclcpp/rclcpp.hpp>
#include <realtime_tools/realtime_publisher.hpp>

#include <predictive_controller/pose_broadcaster_parameters.hpp>

#include "predictive_controller/visibility_control.h"

using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

namespace predictive_controller
{

class PoseBroadcaster : public controller_interface::ControllerInterface
{
public:
  [[nodiscard]] controller_interface::InterfaceConfiguration
  command_interface_configuration() const override;
  [[nodiscard]] controller_interface::InterfaceConfiguration
  state_interface_configuration() const override;
  controller_interface::return_type update(const rclcpp::Time& time,
                                           const rclcpp::Duration& period) override;
  CallbackReturn on_init() override;
  CallbackReturn on_configure(const rclcpp_lifecycle::State& previous_state) override;
  CallbackReturn on_activate(const rclcpp_lifecycle::State& previous_state) override;
  CallbackReturn on_deactivate(const rclcpp_lifecycle::State& previous_state) override;

private:
  std::shared_ptr<pose_broadcaster::ParamListener> params_listener_;
  pose_broadcaster::Params params_;

  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_publisher_;
  std::shared_ptr<realtime_tools::RealtimePublisher<geometry_msgs::msg::PoseStamped>>
      realtime_pose_publisher_;

  std::string end_effector_frame_;
  int end_effector_frame_id;

  pinocchio::Model model_;
  pinocchio::Data data_;

  const std::unordered_set<std::basic_string<char>> allowed_joint_types = {
      "JointModelRX",   "JointModelRY",   "JointModelRZ",   "JointModelRevoluteUnaligned",
      "JointModelRUBX", "JointModelRUBY", "JointModelRUBZ",
  };
  const std::unordered_set<std::basic_string<char>> continous_joint_types = {
      "JointModelRUBX",
      "JointModelRUBY",
      "JointModelRUBZ",
  };

  Eigen::VectorXd q;

  rclcpp::Duration publish_elapsed_{0, 0};
  rclcpp::Duration publish_interval_{0, 0};
};

} // namespace predictive_controller
