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

#include <predictive_controller/noop_effort_controller.hpp>

#include <exception>
#include <string>

namespace predictive_controller
{

controller_interface::InterfaceConfiguration
NoopEffortController::command_interface_configuration() const
{
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

  for (int i = 1; i <= num_joints; ++i)
  {
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/effort");
  }
  return config;
}

controller_interface::InterfaceConfiguration
NoopEffortController::state_interface_configuration() const
{
  return {};
}

controller_interface::return_type NoopEffortController::update(const rclcpp::Time& /*time*/,
                                                               const rclcpp::Duration& /*period*/)
{
  for (auto& command_interface : command_interfaces_)
  {
    command_interface.set_value(0.0);
  }
  return controller_interface::return_type::OK;
}

CallbackReturn NoopEffortController::on_configure(const rclcpp_lifecycle::State& /*previous_state*/)
{
  arm_id_ = get_node()->get_parameter("arm_id").as_string();
  return CallbackReturn::SUCCESS;
}

CallbackReturn NoopEffortController::on_init()
{
  try
  {
    auto_declare<std::string>("arm_id", "fr3");
  }
  catch (const std::exception& e)
  {
    fprintf(stderr, "Exception thrown during init stage with message: %s \n", e.what());
    return CallbackReturn::ERROR;
  }
  return CallbackReturn::SUCCESS;
}

} // namespace predictive_controller

#include "pluginlib/class_list_macros.hpp"
// NOLINTNEXTLINE
PLUGINLIB_EXPORT_CLASS(predictive_controller::NoopEffortController,
                       controller_interface::ControllerInterface)
