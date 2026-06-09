// Minimal MuJoCo viewer extracted from mujoco_ros2_control (MIT License, 2025 Sangtaek Lee).
// Provides a GLFW window and camera controls for live MuJoCo rendering.
#pragma once

#include <mutex>

#include "GLFW/glfw3.h"
#include "mujoco/mujoco.h"

namespace mujoco_sim
{
class MujocoRendering
{
public:
  MujocoRendering(const MujocoRendering&) = delete;
  MujocoRendering& operator=(const MujocoRendering&) = delete;

  static MujocoRendering* getInstance();

  void init(mjModel* mujoco_model, mjData* mujoco_data, std::mutex* state_mutex);
  bool isCloseRequested() const;
  void update();
  void close();

private:
  MujocoRendering();

  static void keyboardCallback(GLFWwindow* window, int key, int scancode, int act, int mods);
  static void mouseButtonCallback(GLFWwindow* window, int button, int act, int mods);
  static void mouseMoveCallback(GLFWwindow* window, double xpos, double ypos);
  static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);

  void keyboardCallbackImpl(GLFWwindow* window, int key, int scancode, int act, int mods);
  void mouseButtonCallbackImpl(GLFWwindow* window, int button, int act, int mods);
  void mouseMoveCallbackImpl(GLFWwindow* window, double xpos, double ypos);
  void scrollCallbackImpl(GLFWwindow* window, double xoffset, double yoffset);

  static MujocoRendering* instance_;

  mjModel* mj_model_;
  mjData* mj_data_;
  std::mutex* state_mutex_{nullptr};

  GLFWwindow* window_{nullptr};
  mjvCamera mjv_cam_;
  mjvOption mjv_opt_;
  mjvScene mjv_scn_;
  mjrContext mjr_con_;
  mjvPerturb mjv_pert_;

  bool button_left_;
  bool button_middle_;
  bool button_right_;
  double lastx_;
  double lasty_;

  bool show_collision_{false};
};
} // namespace mujoco_sim
