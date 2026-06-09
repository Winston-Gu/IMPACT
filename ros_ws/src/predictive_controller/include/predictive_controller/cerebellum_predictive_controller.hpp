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

#include <string>
#include <unordered_set>

#include <Eigen/Dense>
#include <controller_interface/controller_interface.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/wrench_stamped.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/spatial/se3.hpp>
#include <rclcpp/rclcpp.hpp>
#include <realtime_tools/realtime_buffer.hpp>
#include <realtime_tools/realtime_publisher.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float64.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

#include <predictive_controller/cerebellum_predictive_controller_parameters.hpp>

#include "predictive_controller/visibility_control.h"

using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

namespace predictive_controller
{

class CerebellumPredictiveController : public controller_interface::ControllerInterface
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
  void setStiffnessAndDamping();
  void parse_target_pose_();
  void parse_target_joint_();
  void parse_target_wrench_();
  void update_bias_state(double dt, const Eigen::Matrix<double, 6, 1>& measurement);
  double compute_gate(double dt);

  void log_debug_info(const rclcpp::Time& time);
  bool check_topic_publisher_count(const std::string& topic_name);

  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_sub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_sub_;
  rclcpp::Subscription<geometry_msgs::msg::WrenchStamped>::SharedPtr wrench_sub_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr gripper_cmd_sub_;

  bool multiple_publishers_detected_;
  size_t max_allowed_publishers_;

  bool new_target_pose_;
  bool new_target_joint_;
  bool new_target_wrench_;

  realtime_tools::RealtimeBuffer<std::shared_ptr<geometry_msgs::msg::PoseStamped>>
      target_pose_buffer_;
  realtime_tools::RealtimeBuffer<std::shared_ptr<sensor_msgs::msg::JointState>>
      target_joint_buffer_;
  realtime_tools::RealtimeBuffer<std::shared_ptr<geometry_msgs::msg::WrenchStamped>>
      target_wrench_buffer_;
  realtime_tools::RealtimeBuffer<double> gripper_cmd_buffer_{0.0};

  Eigen::Vector3d target_position_;
  Eigen::Quaterniond target_orientation_;
  Eigen::VectorXd target_wrench_;
  pinocchio::SE3 target_pose_;

  std::shared_ptr<cerebellum_predictive_controller::ParamListener> params_listener_;
  cerebellum_predictive_controller::Params params_;

  rclcpp::Publisher<geometry_msgs::msg::WrenchStamped>::SharedPtr wrench_publisher_;
  std::shared_ptr<realtime_tools::RealtimePublisher<geometry_msgs::msg::WrenchStamped>>
      realtime_wrench_publisher_;
  rclcpp::Publisher<geometry_msgs::msg::WrenchStamped>::SharedPtr wrench_slow_pub_;
  rclcpp::Publisher<geometry_msgs::msg::WrenchStamped>::SharedPtr wrench_fast_pub_;
  std::shared_ptr<realtime_tools::RealtimePublisher<geometry_msgs::msg::WrenchStamped>>
      rt_wrench_slow_pub_;
  std::shared_ptr<realtime_tools::RealtimePublisher<geometry_msgs::msg::WrenchStamped>>
      rt_wrench_fast_pub_;
  rclcpp::Publisher<geometry_msgs::msg::WrenchStamped>::SharedPtr applied_wrench_publisher_;
  std::shared_ptr<realtime_tools::RealtimePublisher<geometry_msgs::msg::WrenchStamped>>
      realtime_applied_wrench_publisher_;
  rclcpp::Publisher<geometry_msgs::msg::WrenchStamped>::SharedPtr contact_wrench_publisher_;
  std::shared_ptr<realtime_tools::RealtimePublisher<geometry_msgs::msg::WrenchStamped>>
      realtime_contact_wrench_publisher_;
  rclcpp::Publisher<geometry_msgs::msg::WrenchStamped>::SharedPtr cereb_slow_publisher_;
  rclcpp::Publisher<geometry_msgs::msg::WrenchStamped>::SharedPtr cereb_pred_publisher_;
  rclcpp::Publisher<geometry_msgs::msg::WrenchStamped>::SharedPtr cereb_applied_publisher_;
  rclcpp::Publisher<geometry_msgs::msg::WrenchStamped>::SharedPtr cereb_gate_publisher_;
  rclcpp::Publisher<geometry_msgs::msg::WrenchStamped>::SharedPtr cereb_fast_publisher_;
  std::shared_ptr<realtime_tools::RealtimePublisher<geometry_msgs::msg::WrenchStamped>>
      realtime_cereb_slow_publisher_;
  std::shared_ptr<realtime_tools::RealtimePublisher<geometry_msgs::msg::WrenchStamped>>
      realtime_cereb_pred_publisher_;
  std::shared_ptr<realtime_tools::RealtimePublisher<geometry_msgs::msg::WrenchStamped>>
      realtime_cereb_applied_publisher_;
  std::shared_ptr<realtime_tools::RealtimePublisher<geometry_msgs::msg::WrenchStamped>>
      realtime_cereb_gate_publisher_;
  std::shared_ptr<realtime_tools::RealtimePublisher<geometry_msgs::msg::WrenchStamped>>
      realtime_cereb_fast_publisher_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr cereb_trust_publisher_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr cereb_error_publisher_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr cereb_surprise_filt_publisher_;
  std::shared_ptr<realtime_tools::RealtimePublisher<std_msgs::msg::Bool>>
      realtime_cereb_trust_publisher_;
  std::shared_ptr<realtime_tools::RealtimePublisher<std_msgs::msg::Float64>>
      realtime_cereb_error_publisher_;
  std::shared_ptr<realtime_tools::RealtimePublisher<std_msgs::msg::Float64>>
      realtime_cereb_surprise_filt_publisher_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr cereb_weight_publisher_;
  std::shared_ptr<realtime_tools::RealtimePublisher<std_msgs::msg::Float64>>
      realtime_cereb_weight_publisher_;

  int end_effector_frame_id;

  pinocchio::Model model_;
  pinocchio::Data data_;

  Eigen::MatrixXd stiffness = Eigen::MatrixXd::Zero(6, 6);
  Eigen::MatrixXd damping = Eigen::MatrixXd::Zero(6, 6);

  Eigen::MatrixXd nullspace_stiffness;
  Eigen::MatrixXd nullspace_damping;

  Eigen::VectorXd q;
  Eigen::VectorXd q_pin;
  Eigen::VectorXd dq;
  Eigen::VectorXd q_ref;
  Eigen::VectorXd dq_ref;

  Eigen::VectorXd tau_previous;

  pinocchio::SE3 end_effector_pose;
  pinocchio::Data::Matrix6x J;

  Eigen::VectorXd fp1;
  Eigen::VectorXd fp2;
  Eigen::VectorXd fp3;

  const std::unordered_set<std::basic_string<char>> allowed_joint_types = {
      "JointModelRX",   "JointModelRY",   "JointModelRZ",   "JointModelRevoluteUnaligned",
      "JointModelRUBX", "JointModelRUBY", "JointModelRUBZ",
  };
  const std::unordered_set<std::basic_string<char>> continous_joint_types = {
      "JointModelRUBX",
      "JointModelRUBY",
      "JointModelRUBZ",
  };

  Eigen::VectorXd max_delta_ = Eigen::VectorXd::Zero(6);

  Eigen::MatrixXd nullspace_projection;

  Eigen::VectorXd error = Eigen::VectorXd::Zero(6);

  Eigen::VectorXd tau_task;
  Eigen::VectorXd tau_joint_limits;
  Eigen::VectorXd tau_secondary;
  Eigen::VectorXd tau_nullspace;
  Eigen::VectorXd tau_friction;
  Eigen::VectorXd tau_coriolis;
  Eigen::VectorXd tau_gravity;
  Eigen::VectorXd tau_wrench;
  Eigen::VectorXd tau_meas;
  Eigen::VectorXd tau_model;
  Eigen::VectorXd tau_cereb;
  Eigen::VectorXd tau_ff;
  Eigen::VectorXd tau_d;

  Eigen::VectorXd ddq_hat;
  Eigen::VectorXd dq_prev;
  Eigen::VectorXd residual;

  Eigen::VectorXd w_ext_raw;
  Eigen::VectorXd w_ext_filt;
  Eigen::Matrix<double, 6, 1> w_slow_;
  Eigen::Matrix<double, 6, 1> w_fast_;
  Eigen::Matrix<double, 6, 1> w_slow;
  Eigen::Matrix<double, 6, 1> w_fast;
  Eigen::Matrix<double, 6, 1> w_pred;
  Eigen::Matrix<double, 6, 1> w_ff;
  Eigen::Matrix<double, 6, 1> w_slow_prev_ee_;
  double cereb_weight_est_{0.0};
  double cereb_surprise_{0.0};
  double cereb_surprise_filt_{0.0};
  bool holding_object_{false};
  bool holding_object_prev_{false};
  double holdoff_elapsed_{0.0};
  bool holdoff_checked_{false};
  Eigen::Matrix<double, 6, 1> cereb_gate_;
  double cereb_err_ema{0.0};
  size_t cereb_warmup_count{0};
  size_t cereb_update_counter_{0};

  static constexpr int kCerebPhiDim = 13;
  Eigen::Matrix<double, kCerebPhiDim, 1> cereb_phi_;
  Eigen::Matrix<double, kCerebPhiDim, 1> cereb_k_;
  Eigen::Matrix<double, kCerebPhiDim, kCerebPhiDim> cereb_P_;
  Eigen::Matrix<double, 6, kCerebPhiDim> cereb_W_;
  Eigen::Matrix<double, kCerebPhiDim, kCerebPhiDim> cereb_I_;
  Eigen::Matrix<double, kCerebPhiDim, 1> cereb_P_phi_;
  Eigen::Matrix<double, kCerebPhiDim, kCerebPhiDim> cereb_k_phiT_;
  Eigen::Matrix<double, 6, 1> cereb_e_;
  Eigen::Matrix<double, 6, kCerebPhiDim> cereb_grad_W_;

  Eigen::Matrix<double, 6, 6> observer_A_;
  Eigen::Matrix<double, 6, 6> observer_I6_;
  Eigen::VectorXd observer_Jr_;

  size_t observer_cycle_count{0};
};

} // namespace predictive_controller
