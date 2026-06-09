#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include <mujoco/mujoco.h>
#include <rclcpp/rclcpp.hpp>

namespace mujoco_sim
{
class MuJoCoSimulator
{
private:
  MuJoCoSimulator();

  // Lock the mutex for these calls
  void syncStates();

public:
  // Modern singleton approach
  MuJoCoSimulator(const MuJoCoSimulator&) = delete;
  MuJoCoSimulator& operator=(const MuJoCoSimulator&) = delete;
  MuJoCoSimulator(MuJoCoSimulator&&) = delete;
  MuJoCoSimulator& operator=(MuJoCoSimulator&&) = delete;

  // Use this in ROS2 code
  static MuJoCoSimulator& getInstance()
  {
    static MuJoCoSimulator simulator;
    return simulator;
  }

  // MuJoCo data structures
  mjModel* m = NULL; // MuJoCo model
  mjData* d = NULL;  // MuJoCo data

  // Buffers for data exchange with ROS2-control
  std::vector<double> eff_cmd;
  std::vector<double> gripper_eff_cmd;
  std::vector<double> pos_state;
  std::vector<double> vel_state;
  std::vector<double> eff_state;

  size_t arm_dofs_ = 7;

  // Safety guards for buffers
  std::mutex state_mutex;
  std::mutex command_mutex;

  // Initialization tracking
  bool is_initialized = false;
  std::atomic<bool> controller_ready{false};
  std::mutex init_mutex;
  std::condition_variable init_cv;
  std::mutex controller_mutex;
  std::condition_variable controller_cv;

  // Control input callback for the solver
  static void controlCB(const mjModel* m, mjData* d);
  void controlCBImpl(const mjModel* m, mjData* d);
  void controlCBImplTorque(const mjModel* m, mjData* d);

  // Call this in a separate thread
  static int simulate(const std::string& model_xml);
  int simulateImpl(const std::string& model_xml);

  // Allow external code to mark controllers as ready.
  void setControllerReady(bool ready);
  void setGripperJointNames(const std::string& joint1, const std::string& joint2);
  void setGripperTargetWidth(double width);
  void setGripperGains(double kp, double kd, double max_force);
  bool getBodyMass(const std::string& body_name, double& mass);
  bool getBodyPose(const std::string& body_name, std::array<double, 3>& pos,
                   std::array<double, 4>& quat);
  bool resetSceneFromEnv();
  bool requestSceneReset(double timeout_sec);

  // Non-blocking
  void read(std::vector<double>& pos, std::vector<double>& vel, std::vector<double>& eff);
  void write(const std::vector<double>& eff);

private:
  void applyGripperTarget();
  void resolveGripperJointIds();
  void resolveResetIds();
  bool setMocapPose(int body_id, const std::array<double, 3>& pos,
                    const std::array<double, 4>& quat);

  std::mutex gripper_mutex_;
  std::string gripper_joint1_;
  std::string gripper_joint2_;
  int gripper_joint1_id_ = -1;
  int gripper_joint2_id_ = -1;
  int gripper_qposadr1_ = -1;
  int gripper_qposadr2_ = -1;
  int gripper_qveladr1_ = -1;
  int gripper_qveladr2_ = -1;
  int gripper_act1_id_ = -1;
  int gripper_act2_id_ = -1;
  double gripper_target_width_ = 0.0;
  bool gripper_target_valid_ = false;
  double gripper_kp_ = 200.0;
  double gripper_kd_ = 5.0;
  double gripper_max_force_ = 70.0;

  int cube_anchor_bid_ = -1;
  int basket_anchor_bid_ = -1;
  int cube_weld_eqid_ = -1;
  int basket_weld_eqid_ = -1;
  int cube_free_jid_ = -1;
  int basket_free_jid_ = -1;
  int reset_hold_steps_ = 50;
  int reset_steps_remaining_ = 0;

  std::atomic<uint64_t> reset_counter_{0};
  std::atomic<bool> reset_requested_{false};
  std::atomic<bool> reset_in_progress_{false};
  std::atomic<bool> reset_done_{false};
  std::atomic<bool> sim_loop_started_{false};
  std::mutex reset_mutex_;
  std::condition_variable reset_cv_;
};

} // namespace mujoco_sim
