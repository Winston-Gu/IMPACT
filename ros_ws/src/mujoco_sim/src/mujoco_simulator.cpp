#include "mujoco_sim/mujoco_simulator.h"

#include <chrono>
#include <random>

#include <GLFW/glfw3.h>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <ostream>
#include <rclcpp/duration.hpp>
#include <rclcpp/logger.hpp>
#include <rclcpp/logging.hpp>
#include <sstream>
#include <thread>

#include "mujoco_sim/mujoco_rendering.h"

// ANSI color codes for terminal output
#define GREEN "\033[0;32m"
#define YELLOW "\033[1;33m"
#define RED "\033[0;31m"
#define NC "\033[0m" // No Color

namespace mujoco_sim
{
MuJoCoSimulator::MuJoCoSimulator()
{
}

namespace
{
void render_loop(const mjModel* m, mjData* d, std::mutex* state_mutex)
{
  if (m == nullptr || d == nullptr || state_mutex == nullptr)
  {
    return;
  }

  const char* display = std::getenv("DISPLAY");
  if (display == nullptr || std::strlen(display) == 0)
  {
    std::cerr << "[MuJoCoSimulator] DISPLAY not set; skipping GUI." << std::endl;
    return;
  }

  if (!glfwInit())
  {
    std::cerr << "[MuJoCoSimulator] Failed to initialize GLFW for viewer." << std::endl;
    return;
  }

  auto* renderer = MujocoRendering::getInstance();
  renderer->init(const_cast<mjModel*>(m), d, state_mutex);

  while (!renderer->isCloseRequested())
  {
    renderer->update();
  }

  renderer->close();
#if !defined(__APPLE__) && !defined(_WIN32)
  glfwTerminate();
#endif
}
} // namespace

void MuJoCoSimulator::controlCB(const mjModel* m, mjData* d)
{
  getInstance().controlCBImplTorque(m, d);
}

void MuJoCoSimulator::controlCBImplTorque([[maybe_unused]] const mjModel* m, mjData* d)
{
  command_mutex.lock();

  // Hold torque at zero until controllers signal readiness.
  const size_t ctrl_size = (m != nullptr) ? static_cast<size_t>(m->nu) : eff_cmd.size();
  std::fill(d->ctrl, d->ctrl + ctrl_size, 0.0);
  if (!controller_ready.load(std::memory_order_relaxed))
  {
  }
  else
  {
    const size_t count = std::min(arm_dofs_, eff_cmd.size());
    for (size_t i = 0; i < count; ++i)
    {
      d->ctrl[i] = eff_cmd[i]; // torque control
    }
  }
  if (!gripper_eff_cmd.empty())
  {
    size_t count = std::min(gripper_eff_cmd.size(), ctrl_size);
    for (size_t i = 0; i < count; ++i)
    {
      d->ctrl[i] += gripper_eff_cmd[i];
    }
  }
  command_mutex.unlock();
}

void MuJoCoSimulator::setGripperJointNames(const std::string& joint1, const std::string& joint2)
{
  std::lock_guard<std::mutex> lock(gripper_mutex_);
  gripper_joint1_ = joint1;
  gripper_joint2_ = joint2;
  gripper_joint1_id_ = -1;
  gripper_joint2_id_ = -1;
  gripper_qposadr1_ = -1;
  gripper_qposadr2_ = -1;
  gripper_qveladr1_ = -1;
  gripper_qveladr2_ = -1;
  gripper_act1_id_ = -1;
  gripper_act2_id_ = -1;
}

void MuJoCoSimulator::setGripperTargetWidth(double width)
{
  std::lock_guard<std::mutex> lock(gripper_mutex_);
  gripper_target_width_ = width;
  gripper_target_valid_ = true;
}

void MuJoCoSimulator::setGripperGains(double kp, double kd, double max_force)
{
  std::lock_guard<std::mutex> lock(gripper_mutex_);
  gripper_kp_ = kp;
  gripper_kd_ = kd;
  gripper_max_force_ = max_force;
}

bool MuJoCoSimulator::getBodyPose(const std::string& body_name, std::array<double, 3>& pos,
                                  std::array<double, 4>& quat)
{
  if (m == nullptr || d == nullptr)
  {
    return false;
  }
  int body_id = mj_name2id(m, mjOBJ_BODY, body_name.c_str());
  if (body_id < 0)
  {
    return false;
  }
  std::lock_guard<std::mutex> lock(state_mutex);
  mju_copy3(pos.data(), d->xpos + 3 * body_id);
  mju_copy4(quat.data(), d->xquat + 4 * body_id);
  return true;
}

bool MuJoCoSimulator::getBodyMass(const std::string& body_name, double& mass)
{
  if (m == nullptr)
  {
    return false;
  }
  int body_id = mj_name2id(m, mjOBJ_BODY, body_name.c_str());
  if (body_id < 0)
  {
    return false;
  }
  std::lock_guard<std::mutex> lock(state_mutex);
  mass = m->body_mass[body_id];
  return true;
}

bool MuJoCoSimulator::resetSceneFromEnv()
{
  {
    std::unique_lock<std::mutex> lock(init_mutex);
    if (!is_initialized)
    {
      init_cv.wait(lock, [this] { return is_initialized; });
    }
  }
  if (m == nullptr || d == nullptr)
  {
    return false;
  }

  resolveResetIds();

  auto env_double = [](const char* name, double fallback)
  {
    const char* value = std::getenv(name);
    if (!value)
    {
      return fallback;
    }
    char* end = nullptr;
    double parsed = std::strtod(value, &end);
    if (end == value)
    {
      return fallback;
    }
    return parsed;
  };

  {
    std::lock_guard<std::mutex> cmd_lock(command_mutex);
    std::fill(eff_cmd.begin(), eff_cmd.end(), 0.0);
    std::fill(gripper_eff_cmd.begin(), gripper_eff_cmd.end(), 0.0);
  }

  {
    std::lock_guard<std::mutex> state_lock(state_mutex);
    // Reset full joint configuration to the model keyframe before scene objects.
    if (m->key_qpos != nullptr && m->nq > 0)
    {
      mju_copy(d->qpos, m->key_qpos, m->nq);
    }
    if (m->nv > 0)
    {
      mju_zero(d->qvel, m->nv);
    }

    const double table_surface_z = 0.53;
    const double cube_half = 0.03;
    const double cube_z =
        table_surface_z + cube_half + env_double("MUJOCO_CUBE_SPAWN_CLEARANCE", 0.005);

    std::array<double, 3> cube_pos{{0.45, 0.0, cube_z}};
    std::array<double, 3> basket_pos{{0.65, 0.2, 0.53}};
    std::array<double, 4> cube_quat{{1.0, 0.0, 0.0, 0.0}};
    std::array<double, 4> identity_quat{{1.0, 0.0, 0.0, 0.0}};

    if (cube_anchor_bid_ >= 0)
    {
      cube_pos[0] = m->body_pos[3 * cube_anchor_bid_ + 0];
      cube_pos[1] = m->body_pos[3 * cube_anchor_bid_ + 1];
      cube_pos[2] = m->body_pos[3 * cube_anchor_bid_ + 2];
      cube_quat[0] = m->body_quat[4 * cube_anchor_bid_ + 0];
      cube_quat[1] = m->body_quat[4 * cube_anchor_bid_ + 1];
      cube_quat[2] = m->body_quat[4 * cube_anchor_bid_ + 2];
      cube_quat[3] = m->body_quat[4 * cube_anchor_bid_ + 3];
    }
    if (basket_anchor_bid_ >= 0)
    {
      basket_pos[0] = m->body_pos[3 * basket_anchor_bid_ + 0];
      basket_pos[1] = m->body_pos[3 * basket_anchor_bid_ + 1];
      basket_pos[2] = m->body_pos[3 * basket_anchor_bid_ + 2];
    }

    const char* randomize_env = std::getenv("MUJOCO_RANDOMIZE_SCENE");
    if (randomize_env != nullptr && std::string(randomize_env) != "0")
    {
      std::random_device rd;
      unsigned int seed = rd();
      if (const char* seed_env = std::getenv("MUJOCO_RANDOM_SEED"))
      {
        seed = static_cast<unsigned int>(std::strtoul(seed_env, nullptr, 10));
        seed += static_cast<unsigned int>(reset_counter_.fetch_add(1));
      }
      std::mt19937 rng(seed);

      auto uniform = [&](double lo, double hi)
      {
        std::uniform_real_distribution<double> dist(lo, hi);
        return dist(rng);
      };

      const double cube_x_min = env_double("MUJOCO_CUBE_X_MIN", 0.35);
      const double cube_x_max = env_double("MUJOCO_CUBE_X_MAX", 0.6);
      const double cube_y_min = env_double("MUJOCO_CUBE_Y_MIN", -0.2);
      const double cube_y_max = env_double("MUJOCO_CUBE_Y_MAX", 0.2);
      const double cube_mass_min = env_double("MUJOCO_CUBE_MASS_MIN", 0.05);
      const double cube_mass_max = env_double("MUJOCO_CUBE_MASS_MAX", 0.2);
      const double basket_x_min = env_double("MUJOCO_BASKET_X_MIN", 0.6);
      const double basket_x_max = env_double("MUJOCO_BASKET_X_MAX", 0.85);
      const double basket_y_min = env_double("MUJOCO_BASKET_Y_MIN", 0.1);
      const double basket_y_max = env_double("MUJOCO_BASKET_Y_MAX", 0.35);

      cube_pos[0] = uniform(cube_x_min, cube_x_max);
      cube_pos[1] = uniform(cube_y_min, cube_y_max);
      cube_pos[2] = cube_z;
      basket_pos[0] = uniform(basket_x_min, basket_x_max);
      basket_pos[1] = uniform(basket_y_min, basket_y_max);
      basket_pos[2] = table_surface_z;
      const double cube_mass = uniform(cube_mass_min, cube_mass_max);

      int cube_bid = mj_name2id(m, mjOBJ_BODY, "pick_cube");
      if (cube_bid >= 0)
      {
        double old_mass = m->body_mass[cube_bid];
        if (old_mass > 0.0)
        {
          double scale = cube_mass / old_mass;
          m->body_mass[cube_bid] = cube_mass;
          for (int i = 0; i < 3; ++i)
          {
            m->body_inertia[3 * cube_bid + i] *= scale;
          }
        }
      }

      auto logger = rclcpp::get_logger("MuJoCoSimulator");
      RCLCPP_INFO(logger,
                  "Randomized scene seed=%u cube=(%.3f, %.3f, %.3f) mass=%.3f basket=(%.3f, %.3f, "
                  "%.3f)",
                  seed, cube_pos[0], cube_pos[1], cube_pos[2], cube_mass, basket_pos[0],
                  basket_pos[1], basket_pos[2]);
    }

    setMocapPose(cube_anchor_bid_, cube_pos, cube_quat);
    setMocapPose(basket_anchor_bid_, basket_pos, identity_quat);

    if (cube_free_jid_ >= 0)
    {
      int qpos_adr = m->jnt_qposadr[cube_free_jid_];
      if (qpos_adr + 6 < m->nq)
      {
        d->qpos[qpos_adr + 0] = cube_pos[0];
        d->qpos[qpos_adr + 1] = cube_pos[1];
        d->qpos[qpos_adr + 2] = cube_pos[2];
        d->qpos[qpos_adr + 3] = cube_quat[0];
        d->qpos[qpos_adr + 4] = cube_quat[1];
        d->qpos[qpos_adr + 5] = cube_quat[2];
        d->qpos[qpos_adr + 6] = cube_quat[3];
      }

      int qvel_adr = m->jnt_dofadr[cube_free_jid_];
      if (qvel_adr + 5 < m->nv)
      {
        for (int i = 0; i < 6; ++i)
        {
          d->qvel[qvel_adr + i] = 0.0;
        }
      }
    }

    if (basket_free_jid_ >= 0)
    {
      int qpos_adr = m->jnt_qposadr[basket_free_jid_];
      if (qpos_adr + 6 < m->nq)
      {
        d->qpos[qpos_adr + 0] = basket_pos[0];
        d->qpos[qpos_adr + 1] = basket_pos[1];
        d->qpos[qpos_adr + 2] = basket_pos[2];
        d->qpos[qpos_adr + 3] = 1.0;
        d->qpos[qpos_adr + 4] = 0.0;
        d->qpos[qpos_adr + 5] = 0.0;
        d->qpos[qpos_adr + 6] = 0.0;
      }

      int qvel_adr = m->jnt_dofadr[basket_free_jid_];
      if (qvel_adr + 5 < m->nv)
      {
        for (int i = 0; i < 6; ++i)
        {
          d->qvel[qvel_adr + i] = 0.0;
        }
      }
    }

    if (cube_weld_eqid_ >= 0 && cube_weld_eqid_ < m->neq)
    {
      d->eq_active[cube_weld_eqid_] = 1;
    }
    if (basket_weld_eqid_ >= 0 && basket_weld_eqid_ < m->neq)
    {
      d->eq_active[basket_weld_eqid_] = 1;
    }
    reset_steps_remaining_ = reset_hold_steps_;

    mj_forward(m, d);
    syncStates();
  }

  return true;
}

bool MuJoCoSimulator::requestSceneReset(double timeout_sec)
{
  {
    std::unique_lock<std::mutex> lock(init_mutex);
    if (!is_initialized)
    {
      init_cv.wait(lock, [this] { return is_initialized; });
    }
  }

  if (!sim_loop_started_.load(std::memory_order_relaxed) ||
      !controller_ready.load(std::memory_order_relaxed))
  {
    return resetSceneFromEnv();
  }

  reset_done_.store(false, std::memory_order_relaxed);
  reset_requested_.store(true, std::memory_order_relaxed);
  std::unique_lock<std::mutex> lock(reset_mutex_);
  if (timeout_sec <= 0.0)
  {
    reset_cv_.wait(lock, [this] { return reset_done_.load(std::memory_order_relaxed); });
    return true;
  }
  return reset_cv_.wait_for(lock, std::chrono::duration<double>(timeout_sec),
                            [this] { return reset_done_.load(std::memory_order_relaxed); });
}

void MuJoCoSimulator::resolveGripperJointIds()
{
  if (m == nullptr)
  {
    return;
  }
  std::lock_guard<std::mutex> lock(gripper_mutex_);
  if (gripper_joint1_.empty() || gripper_joint2_.empty())
  {
    return;
  }

  gripper_joint1_id_ = mj_name2id(m, mjOBJ_JOINT, gripper_joint1_.c_str());
  gripper_joint2_id_ = mj_name2id(m, mjOBJ_JOINT, gripper_joint2_.c_str());
  if (gripper_joint1_id_ < 0 || gripper_joint2_id_ < 0)
  {
    return;
  }
  gripper_qposadr1_ = m->jnt_qposadr[gripper_joint1_id_];
  gripper_qposadr2_ = m->jnt_qposadr[gripper_joint2_id_];
  gripper_qveladr1_ = m->jnt_dofadr[gripper_joint1_id_];
  gripper_qveladr2_ = m->jnt_dofadr[gripper_joint2_id_];
  gripper_act1_id_ = -1;
  gripper_act2_id_ = -1;
  for (int act_id = 0; act_id < m->nu; ++act_id)
  {
    if (m->actuator_trntype[act_id] != mjTRN_JOINT)
    {
      continue;
    }
    int joint_id = m->actuator_trnid[2 * act_id];
    if (joint_id == gripper_joint1_id_)
    {
      gripper_act1_id_ = act_id;
    }
    else if (joint_id == gripper_joint2_id_)
    {
      gripper_act2_id_ = act_id;
    }
  }
}

void MuJoCoSimulator::resolveResetIds()
{
  if (m == nullptr)
  {
    return;
  }
  cube_anchor_bid_ = mj_name2id(m, mjOBJ_BODY, "pick_cube_anchor");
  basket_anchor_bid_ = mj_name2id(m, mjOBJ_BODY, "target_basket_anchor");
  cube_weld_eqid_ = mj_name2id(m, mjOBJ_EQUALITY, "pick_cube_anchor_weld");
  basket_weld_eqid_ = mj_name2id(m, mjOBJ_EQUALITY, "basket_anchor_weld");
  cube_free_jid_ = mj_name2id(m, mjOBJ_JOINT, "pick_cube_free");
  basket_free_jid_ = mj_name2id(m, mjOBJ_JOINT, "target_basket_free");
}

bool MuJoCoSimulator::setMocapPose(int body_id, const std::array<double, 3>& pos,
                                   const std::array<double, 4>& quat)
{
  if (m == nullptr || d == nullptr)
  {
    return false;
  }
  if (body_id < 0 || body_id >= m->nbody)
  {
    return false;
  }
  int mocap_id = m->body_mocapid[body_id];
  if (mocap_id < 0 || mocap_id >= m->nmocap)
  {
    return false;
  }
  mju_copy3(d->mocap_pos + 3 * mocap_id, pos.data());
  mju_copy4(d->mocap_quat + 4 * mocap_id, quat.data());
  return true;
}

void MuJoCoSimulator::applyGripperTarget()
{
  if (m == nullptr || d == nullptr)
  {
    return;
  }
  double target_width = 0.0;
  int qposadr1 = -1;
  int qposadr2 = -1;
  int qveladr1 = -1;
  int qveladr2 = -1;
  int act1 = -1;
  int act2 = -1;
  int jid1 = -1;
  int jid2 = -1;
  double kp = 0.0;
  double kd = 0.0;
  double max_force = 0.0;

  {
    std::lock_guard<std::mutex> lock(gripper_mutex_);
    if (!gripper_target_valid_)
    {
      return;
    }
    target_width = gripper_target_width_;
    jid1 = gripper_joint1_id_;
    jid2 = gripper_joint2_id_;
    qposadr1 = gripper_qposadr1_;
    qposadr2 = gripper_qposadr2_;
    qveladr1 = gripper_qveladr1_;
    qveladr2 = gripper_qveladr2_;
    act1 = gripper_act1_id_;
    act2 = gripper_act2_id_;
    kp = gripper_kp_;
    kd = gripper_kd_;
    max_force = gripper_max_force_;
  }

  if (jid1 < 0 || jid2 < 0 || qposadr1 < 0 || qposadr2 < 0)
  {
    resolveGripperJointIds();
    return;
  }

  if (act1 < 0 || act2 < 0)
  {
    resolveGripperJointIds();
    return;
  }

  double finger_pos = 0.5 * target_width;
  double min1 = m->jnt_range[2 * jid1];
  double max1 = m->jnt_range[2 * jid1 + 1];
  double min2 = m->jnt_range[2 * jid2];
  double max2 = m->jnt_range[2 * jid2 + 1];
  finger_pos = std::max(min1, std::min(max1, finger_pos));
  double finger_pos2 = std::max(min2, std::min(max2, finger_pos));

  if (qposadr1 < m->nq && qposadr2 < m->nq && qveladr1 < m->nv && qveladr2 < m->nv)
  {
    double err1 = finger_pos - d->qpos[qposadr1];
    double err2 = finger_pos2 - d->qpos[qposadr2];
    double vel1 = d->qvel[qveladr1];
    double vel2 = d->qvel[qveladr2];
    double force1 = kp * err1 - kd * vel1;
    double force2 = kp * err2 - kd * vel2;
    if (m->actuator_forcelimited != nullptr)
    {
      if (m->actuator_forcelimited[act1])
      {
        double minf = m->actuator_forcerange[2 * act1];
        double maxf = m->actuator_forcerange[2 * act1 + 1];
        force1 = std::max(minf, std::min(maxf, force1));
      }
      if (m->actuator_forcelimited[act2])
      {
        double minf = m->actuator_forcerange[2 * act2];
        double maxf = m->actuator_forcerange[2 * act2 + 1];
        force2 = std::max(minf, std::min(maxf, force2));
      }
    }
    force1 = std::max(-max_force, std::min(max_force, force1));
    force2 = std::max(-max_force, std::min(max_force, force2));
    std::lock_guard<std::mutex> lock(command_mutex);
    if (act1 >= 0 && act1 < static_cast<int>(gripper_eff_cmd.size()))
    {
      gripper_eff_cmd[act1] = force1;
    }
    if (act2 >= 0 && act2 < static_cast<int>(gripper_eff_cmd.size()))
    {
      gripper_eff_cmd[act2] = force2;
    }
  }
}

int MuJoCoSimulator::simulate(const std::string& model_xml)
{
  return getInstance().simulateImpl(model_xml);
}

int MuJoCoSimulator::simulateImpl(const std::string& model_xml)
{
  // Make sure that the ROS2-control system_interface only gets valid data in
  // read(). We lock until we are done with simulation setup.
  state_mutex.lock();
  rclcpp::Logger logger = rclcpp::get_logger("MuJoCoSimulator");

  // load and compile model
  char error[1000] = "Could not load binary model";
  m = mj_loadXML(model_xml.c_str(), nullptr, error, 1000);
  if (!m)
  {
    RCLCPP_ERROR_STREAM(logger, "Could not start the simulation: " << error);
    /*mju_error_s("Load model error: %s", error);*/
    return 1;
  }
  arm_dofs_ = std::min<size_t>(7, static_cast<size_t>(m->nu));

  RCLCPP_INFO_STREAM(logger, GREEN << "Creating data for simulation" << NC);
  // Set initial state with the keyframe mechanism from xml
  d = mj_makeData(m);
  mju_copy(d->qpos, m->key_qpos, m->nq);
  RCLCPP_INFO(logger, "Initial qpos from keyframe:");
  for (int i = 0; i < m->nq; ++i)
  {
    RCLCPP_INFO(logger, "  qpos[%d] = %.6f", i, d->qpos[i]);
  }

  const char* randomize_env = std::getenv("MUJOCO_RANDOMIZE_SCENE");
  if (randomize_env != nullptr && std::string(randomize_env) != "0")
  {
    auto env_double = [](const char* name, double fallback)
    {
      const char* value = std::getenv(name);
      if (!value)
      {
        return fallback;
      }
      char* end = nullptr;
      double parsed = std::strtod(value, &end);
      if (end == value)
      {
        return fallback;
      }
      return parsed;
    };

    std::random_device rd;
    unsigned int seed = rd();
    if (const char* seed_env = std::getenv("MUJOCO_RANDOM_SEED"))
    {
      seed = static_cast<unsigned int>(std::strtoul(seed_env, nullptr, 10));
    }
    std::mt19937 rng(seed);

    auto uniform = [&](double lo, double hi)
    {
      std::uniform_real_distribution<double> dist(lo, hi);
      return dist(rng);
    };

    const double table_surface_z = 0.53;
    const double cube_half = 0.03;
    const double cube_z =
        table_surface_z + cube_half + env_double("MUJOCO_CUBE_SPAWN_CLEARANCE", 0.005);

    const double cube_x_min = env_double("MUJOCO_CUBE_X_MIN", 0.35);
    const double cube_x_max = env_double("MUJOCO_CUBE_X_MAX", 0.6);
    const double cube_y_min = env_double("MUJOCO_CUBE_Y_MIN", -0.2);
    const double cube_y_max = env_double("MUJOCO_CUBE_Y_MAX", 0.2);
    const double cube_mass_min = env_double("MUJOCO_CUBE_MASS_MIN", 0.05);
    const double cube_mass_max = env_double("MUJOCO_CUBE_MASS_MAX", 0.2);
    const double basket_x_min = env_double("MUJOCO_BASKET_X_MIN", 0.6);
    const double basket_x_max = env_double("MUJOCO_BASKET_X_MAX", 0.85);
    const double basket_y_min = env_double("MUJOCO_BASKET_Y_MIN", 0.1);
    const double basket_y_max = env_double("MUJOCO_BASKET_Y_MAX", 0.35);

    const double cube_x = uniform(cube_x_min, cube_x_max);
    const double cube_y = uniform(cube_y_min, cube_y_max);
    const double basket_x = uniform(basket_x_min, basket_x_max);
    const double basket_y = uniform(basket_y_min, basket_y_max);
    const double cube_mass = uniform(cube_mass_min, cube_mass_max);

    int cube_jid = mj_name2id(m, mjOBJ_JOINT, "pick_cube_free");
    if (cube_jid >= 0)
    {
      int qpos_adr = m->jnt_qposadr[cube_jid];
      if (qpos_adr + 6 < m->nq)
      {
        d->qpos[qpos_adr + 0] = cube_x;
        d->qpos[qpos_adr + 1] = cube_y;
        d->qpos[qpos_adr + 2] = cube_z;
        d->qpos[qpos_adr + 3] = 1.0;
        d->qpos[qpos_adr + 4] = 0.0;
        d->qpos[qpos_adr + 5] = 0.0;
        d->qpos[qpos_adr + 6] = 0.0;
      }

      int qvel_adr = m->jnt_dofadr[cube_jid];
      if (qvel_adr + 5 < m->nv)
      {
        for (int i = 0; i < 6; ++i)
        {
          d->qvel[qvel_adr + i] = 0.0;
        }
      }
    }

    int cube_bid = mj_name2id(m, mjOBJ_BODY, "pick_cube");
    if (cube_bid >= 0)
    {
      double old_mass = m->body_mass[cube_bid];
      if (old_mass > 0.0)
      {
        double scale = cube_mass / old_mass;
        m->body_mass[cube_bid] = cube_mass;
        for (int i = 0; i < 3; ++i)
        {
          m->body_inertia[3 * cube_bid + i] *= scale;
        }
      }
    }

    int basket_bid = mj_name2id(m, mjOBJ_BODY, "target_basket");
    if (basket_bid >= 0)
    {
      m->body_pos[3 * basket_bid + 0] = basket_x;
      m->body_pos[3 * basket_bid + 1] = basket_y;
      m->body_pos[3 * basket_bid + 2] = 0.53;
    }

    mj_forward(m, d);

    RCLCPP_INFO(
        logger,
        "Randomized scene seed=%u cube=(%.3f, %.3f, %.3f) mass=%.3f basket=(%.3f, %.3f, %.3f)",
        seed, cube_x, cube_y, cube_z, cube_mass, basket_x, basket_y, 0.53);
  }

  // Initialize buffers for ROS2-control.
  pos_state.resize(arm_dofs_);
  vel_state.resize(arm_dofs_);
  eff_state.resize(arm_dofs_);

  eff_cmd.resize(arm_dofs_);
  gripper_eff_cmd.resize(m->nu, 0.0);

  RCLCPP_INFO_STREAM(logger, GREEN << "Syncing states." << NC);
  syncStates();
  state_mutex.unlock();

  // Mark initialization as complete and notify waiting threads
  {
    std::lock_guard<std::mutex> lock(init_mutex);
    is_initialized = true;
  }
  init_cv.notify_all();
  RCLCPP_INFO_STREAM(logger, GREEN << "✓ MuJoCo initialization complete - ready for control" << NC);

  // Connect our specific control input callback for MuJoCo's engine.
  mjcb_control = MuJoCoSimulator::controlCB;

  bool viewer_enabled = (std::getenv("MUJOCO_SIM_GUI") != nullptr);
  // Optional live viewer that allows mouse perturbations. Enable with MUJOCO_SIM_GUI=1.
  if (viewer_enabled)
  {
    std::thread viewer(render_loop, m, d, &state_mutex);
    viewer.detach();
    RCLCPP_INFO(logger, "Started interactive MuJoCo GUI (MUJOCO_SIM_GUI=1)");
  }

  rclcpp::Clock::SharedPtr clock = std::make_shared<rclcpp::Clock>(RCL_ROS_TIME);

  RCLCPP_INFO_STREAM(logger, GREEN << "Starting simulation" << NC);
  /*auto previous_time = clock->now();*/
  /*auto dt = (double)m->opt.timestep;*/

  auto starting_time = clock->now();

  // Wait for external controllers to signal readiness before stepping physics.
  {
    std::unique_lock<std::mutex> lock(controller_mutex);
    controller_cv.wait(lock, [this] { return controller_ready.load(std::memory_order_relaxed); });
  }

  // Simulate in realtime
  sim_loop_started_.store(true, std::memory_order_relaxed);
  while (true)
  {
    if (reset_requested_.load(std::memory_order_relaxed) &&
        !reset_in_progress_.load(std::memory_order_relaxed))
    {
      reset_in_progress_.store(true, std::memory_order_relaxed);
      reset_requested_.store(false, std::memory_order_relaxed);
      RCLCPP_INFO(logger, "Applying queued scene reset");
      resetSceneFromEnv();
      reset_in_progress_.store(false, std::memory_order_relaxed);
      reset_done_.store(true, std::memory_order_relaxed);
      reset_cv_.notify_all();
    }
    if (reset_steps_remaining_ > 0)
    {
      --reset_steps_remaining_;
      if (reset_steps_remaining_ == 0)
      {
        if (cube_weld_eqid_ >= 0 && cube_weld_eqid_ < m->neq)
        {
          d->eq_active[cube_weld_eqid_] = 0;
        }
        if (basket_weld_eqid_ >= 0 && basket_weld_eqid_ < m->neq)
        {
          d->eq_active[basket_weld_eqid_] = 0;
        }
        RCLCPP_INFO(logger, "Scene reset released anchor welds");
      }
    }

    applyGripperTarget();
    mj_step1(m, d);
    /*mj_step(m, d);*/

    // Provide fresh data for ROS2-control
    state_mutex.lock();
    syncStates();
    RCLCPP_DEBUG_STREAM_THROTTLE(logger, *clock, 1000,
                                 "qpos: " << d->qpos[0] << ", " << d->qpos[1] << ", " << d->qpos[2]
                                          << ", " << d->qpos[3] << ", " << d->qpos[4] << ", "
                                          << d->qpos[5] << ", " << d->qpos[6]
                                          << ", fingers: " << d->qpos[7] << ", " << d->qpos[8]);
    RCLCPP_DEBUG_STREAM_THROTTLE(
        logger, *clock, 1000,
        "Control: " << d->ctrl[0] << ", " << d->ctrl[1] << ", " << d->ctrl[2] << ", " << d->ctrl[3]
                    << ", " << d->ctrl[4] << ", " << d->ctrl[5] << ", " << d->ctrl[6]);
    if (std::any_of(eff_cmd.begin(), eff_cmd.end(), [](double value) { return value > 0.0; }))
    {
      RCLCPP_DEBUG_STREAM_THROTTLE(logger, *clock, 1000,
                                   "Command: " << eff_cmd[0] << ", " << eff_cmd[1] << ", "
                                               << eff_cmd[2] << ", " << eff_cmd[3] << ", "
                                               << eff_cmd[4] << ", " << eff_cmd[5] << ", "
                                               << eff_cmd[6]);
    }

    RCLCPP_DEBUG_STREAM_THROTTLE(logger, *clock, 1000,
                                 "State: " << d->qpos[0] << ", " << d->qpos[1] << ", " << d->qpos[2]
                                           << ", " << d->qpos[3] << ", " << d->qpos[4] << ", "
                                           << d->qpos[5] << ", " << d->qpos[6]);

    state_mutex.unlock();

    // Sync time
    while (clock->now() < starting_time + rclcpp::Duration::from_seconds(d->time))
    {
      rclcpp::sleep_for(100 * rclcpp::nanoseconds(1));
    }

    RCLCPP_DEBUG_STREAM_THROTTLE(logger, *clock, 1000, "Time: " << d->time);

    mj_step2(m, d);
    /*previous_time = clock->now();*/
  }

  return 0;
}

void MuJoCoSimulator::setControllerReady(bool ready)
{
  controller_ready.store(ready, std::memory_order_relaxed);
  controller_cv.notify_all();
}

void MuJoCoSimulator::read(std::vector<double>& pos, std::vector<double>& vel,
                           std::vector<double>& eff)
{
  // Wait for MuJoCo to complete initialization on first call
  {
    std::unique_lock<std::mutex> lock(init_mutex);
    init_cv.wait(lock, [this] { return is_initialized; });
  }

  // Now safely read the state
  if (state_mutex.try_lock())
  {
    auto logger = rclcpp::get_logger("MuJoCoSimulator");
    static bool size_warned = false;

    const auto copy_or_warn =
        [&](const std::vector<double>& src, std::vector<double>& dst, const char* name)
    {
      const auto n = std::min(src.size(), dst.size());
      std::copy_n(src.begin(), n, dst.begin());
      if (src.size() != dst.size() && !size_warned)
      {
        RCLCPP_WARN(logger,
                    "read(): %s size mismatch (src=%zu, dst=%zu); copying first %zu entries.", name,
                    src.size(), dst.size(), n);
      }
    };

    copy_or_warn(pos_state, pos, "pos");
    copy_or_warn(vel_state, vel, "vel");
    copy_or_warn(eff_state, eff, "eff");
    size_warned = true;
    state_mutex.unlock();
  }
  else
  {
    auto logger = rclcpp::get_logger("MuJoCoSimulator");
    RCLCPP_DEBUG(logger, "read(): state_mutex busy, returning previous values");
  }
}

void MuJoCoSimulator::write(const std::vector<double>& eff)
{
  // Wait for initialization so eff_cmd has been sized.
  {
    std::unique_lock<std::mutex> lock(init_mutex);
    init_cv.wait(lock, [this] { return is_initialized; });
  }

  std::lock_guard<std::mutex> lock(command_mutex);
  auto logger = rclcpp::get_logger("MuJoCoSimulator");
  if (eff_cmd.empty())
  {
    RCLCPP_WARN(logger, "write(): simulator not initialized, eff_cmd is empty; dropping command");
    return;
  }
  if (eff.size() < eff_cmd.size())
  {
    RCLCPP_WARN_ONCE(
        logger, "write(): command size %zu < expected %zu, padding remaining actuators with 0.0",
        eff.size(), eff_cmd.size());
    std::copy(eff.begin(), eff.end(), eff_cmd.begin());
    std::fill(eff_cmd.begin() + eff.size(), eff_cmd.end(), 0.0);
  }
  else if (eff.size() > eff_cmd.size())
  {
    RCLCPP_WARN_ONCE(logger, "write(): command size %zu > expected %zu, truncating extra entries",
                     eff.size(), eff_cmd.size());
    std::copy(eff.begin(), eff.begin() + eff_cmd.size(), eff_cmd.begin());
  }
  else
  {
    eff_cmd = eff;
  }

  // Detect when controllers start sending non-zero commands and release the pause.
  const size_t check_len = std::min<size_t>(eff_cmd.size(), arm_dofs_);
  bool any_nonzero = std::any_of(eff_cmd.begin(), eff_cmd.begin() + check_len,
                                 [](double v) { return std::abs(v) > 1e-6; });
  if (any_nonzero && !controller_ready.load(std::memory_order_relaxed))
  {
    controller_ready.store(true, std::memory_order_relaxed);
    controller_cv.notify_all();
  }

  // Keep a copy of the raw (padded/truncated) command for logging before sanitization.
  std::vector<double> raw_cmd = eff_cmd;

  // Sanitize commands to avoid NaN/Inf; do not clamp magnitudes.
  bool bad = false;
  bool nan_or_inf = false;
  for (auto& v : eff_cmd)
  {
    if (!std::isfinite(v))
    {
      bad = true;
      nan_or_inf = true;
      v = 0.0;
      continue;
    }
  }
  if (bad)
  {
    auto logger = rclcpp::get_logger("MuJoCoSimulator");
    static size_t warn_count = 0;
    std::stringstream ss_raw;
    std::stringstream ss_clean;
    size_t show = std::min<size_t>(eff_cmd.size(), 7);
    for (size_t i = 0; i < show; ++i)
    {
      ss_raw << raw_cmd[i] << (i + 1 == show ? "" : ", ");
      ss_clean << eff_cmd[i] << (i + 1 == show ? "" : ", ");
    }
    if (warn_count < 5 || (warn_count % 100 == 0))
    {
      RCLCPP_WARN(logger,
                  "write(): cleaned controller command (nan/inf: %s). "
                  "Raw first %zu: [%s] -> Clean first %zu: [%s]",
                  nan_or_inf ? "yes" : "no", show, ss_raw.str().c_str(), show,
                  ss_clean.str().c_str());
    }
    ++warn_count;
  }
}

void MuJoCoSimulator::syncStates()
{
  for (size_t i = 0; i < arm_dofs_; ++i)
  {
    pos_state[i] = d->qpos[i];
    vel_state[i] = d->qvel[i];
    eff_state[i] = d->qfrc_actuator[i];
  }
}

} // namespace mujoco_sim
