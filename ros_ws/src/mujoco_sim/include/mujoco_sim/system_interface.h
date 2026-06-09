#pragma once

#include <atomic>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <map>
#include <rclcpp/clock.hpp>
#include <rclcpp/executors/single_threaded_executor.hpp>
#include <rclcpp/node.hpp>
#include <thread>

#include "hardware_interface/handle.hpp"
#include "hardware_interface/hardware_info.hpp"
#include "hardware_interface/system.hpp"
#include "hardware_interface/system_interface.hpp"
#include "mujoco_sim/mujoco_simulator.h"
#include "rclcpp/macros.hpp"
#include "rclcpp_lifecycle/node_interfaces/lifecycle_node_interface.hpp"
#include "std_msgs/msg/float64.hpp"
#include "std_srvs/srv/trigger.hpp"

namespace mujoco_sim
{

class MujocoCameraPublisher;

class Simulator : public hardware_interface::SystemInterface
{
public:
  using return_type = hardware_interface::return_type;
  using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

  RCLCPP_SHARED_PTR_DEFINITIONS(Simulator)

  CallbackReturn on_init(const hardware_interface::HardwareInfo& info) override;

  std::vector<hardware_interface::StateInterface> export_state_interfaces() override;
  std::vector<hardware_interface::CommandInterface> export_command_interfaces() override;

  return_type prepare_command_mode_switch(const std::vector<std::string>& start_interfaces,
                                          const std::vector<std::string>& stop_interfaces) override;

  return_type read(const rclcpp::Time& time, const rclcpp::Duration& period) override;
  return_type write(const rclcpp::Time& time, const rclcpp::Duration& period) override;

  // Create a ROS clock instance
  rclcpp::Clock::SharedPtr clock;

private:
  // Command buffers for the controllers
  std::vector<double> m_effort_commands;

  // State buffers for the controllers
  std::vector<double> m_positions;
  std::vector<double> m_velocities;
  std::vector<double> m_efforts;

  // Run MuJoCo's solver in a separate thread
  std::thread m_simulation;
  std::thread gripper_spin_thread_;
  std::atomic<bool> gripper_spin_active_{false};

  // Parameters
  std::string m_mujoco_model;

  rclcpp::Node::SharedPtr gripper_node_;
  rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr gripper_sub_;
  rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr gripper_force_sub_;
  rclcpp::executors::SingleThreadedExecutor gripper_executor_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr reset_scene_srv_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr object_mass_srv_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr cube_pose_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr basket_pose_pub_;
  rclcpp::TimerBase::SharedPtr object_pose_timer_;
  std::shared_ptr<MujocoCameraPublisher> camera_publisher_;
  std::string object_frame_id_;
  std::string cube_body_name_ = "pick_cube";
  double gripper_kp_ = 2000.0;
  double gripper_kd_ = 5.0;
};

} // namespace mujoco_sim
