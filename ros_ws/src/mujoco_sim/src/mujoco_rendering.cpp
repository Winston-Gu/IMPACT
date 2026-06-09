// Copyright (c) 2025 Sangtaek Lee
// MIT License (see upstream mujoco_ros2_control). Minimal integration for mujoco_sim viewer.

#include "mujoco_sim/mujoco_rendering.h"

#include <mutex>

namespace mujoco_sim
{
MujocoRendering* MujocoRendering::instance_ = nullptr;

MujocoRendering* MujocoRendering::getInstance()
{
  if (instance_ == nullptr)
  {
    instance_ = new MujocoRendering();
  }
  return instance_;
}

MujocoRendering::MujocoRendering()
    : mj_model_(nullptr), mj_data_(nullptr), state_mutex_(nullptr), button_left_(false),
      button_middle_(false), button_right_(false), lastx_(0.0), lasty_(0.0)
{
}

void MujocoRendering::init(mjModel* mujoco_model, mjData* mujoco_data, std::mutex* state_mutex)
{
  mj_model_ = mujoco_model;
  mj_data_ = mujoco_data;
  state_mutex_ = state_mutex;

  glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);
  glfwWindowHint(GLFW_DOUBLEBUFFER, GLFW_TRUE);
  window_ = glfwCreateWindow(1200, 900, "MuJoCo Live Sim", nullptr, nullptr);
  glfwMakeContextCurrent(window_);

  mjv_defaultCamera(&mjv_cam_);
  mjv_defaultOption(&mjv_opt_);
  mjv_defaultScene(&mjv_scn_);
  mjr_defaultContext(&mjr_con_);
  mjv_defaultPerturb(&mjv_pert_);

  mjv_opt_.geomgroup[3] = 0;
  mjv_opt_.geomgroup[2] = 1;

  mjv_cam_.type = mjCAMERA_FREE;
  mjv_cam_.distance = 8.;

  mjv_makeScene(mj_model_, &mjv_scn_, 2000);
  mjr_makeContext(mj_model_, &mjr_con_, mjFONTSCALE_150);

  glfwSetKeyCallback(window_, &MujocoRendering::keyboardCallback);
  glfwSetCursorPosCallback(window_, &MujocoRendering::mouseMoveCallback);
  glfwSetMouseButtonCallback(window_, &MujocoRendering::mouseButtonCallback);
  glfwSetScrollCallback(window_, &MujocoRendering::scrollCallback);

  // Avoid vsync-induced stalls when RViz is also running.
  glfwSwapInterval(0);
}

bool MujocoRendering::isCloseRequested() const
{
  return window_ == nullptr ? true : glfwWindowShouldClose(window_);
}

void MujocoRendering::update()
{
  if (window_ == nullptr)
  {
    return;
  }

  std::unique_lock<std::mutex> lock;
  if (state_mutex_ != nullptr)
  {
    lock = std::unique_lock<std::mutex>(*state_mutex_);
  }

  mjrRect viewport = {0, 0, 0, 0};
  glfwGetFramebufferSize(window_, &viewport.width, &viewport.height);
  glfwMakeContextCurrent(window_);

  mjr_setBuffer(mjFB_WINDOW, &mjr_con_);

  mjv_updateScene(mj_model_, mj_data_, &mjv_opt_, &mjv_pert_, &mjv_cam_, mjCAT_ALL, &mjv_scn_);
  mjr_render(viewport, &mjv_scn_, &mjr_con_);

  lock.unlock();

  glfwSwapBuffers(window_);
  glfwPollEvents();
}

void MujocoRendering::close()
{
  mjv_freeScene(&mjv_scn_);
  mjr_freeContext(&mjr_con_);
  if (window_ != nullptr)
  {
    glfwDestroyWindow(window_);
    window_ = nullptr;
  }
#if defined(__APPLE__) || defined(_WIN32)
  glfwTerminate();
#endif
}

void MujocoRendering::keyboardCallback(GLFWwindow* window, int key, int scancode, int act, int mods)
{
  getInstance()->keyboardCallbackImpl(window, key, scancode, act, mods);
}

void MujocoRendering::mouseButtonCallback(GLFWwindow* window, int button, int act, int mods)
{
  getInstance()->mouseButtonCallbackImpl(window, button, act, mods);
}

void MujocoRendering::mouseMoveCallback(GLFWwindow* window, double xpos, double ypos)
{
  getInstance()->mouseMoveCallbackImpl(window, xpos, ypos);
}

void MujocoRendering::scrollCallback(GLFWwindow* window, double xoffset, double yoffset)
{
  getInstance()->scrollCallbackImpl(window, xoffset, yoffset);
}

void MujocoRendering::keyboardCallbackImpl(GLFWwindow* /*window*/, int key, int /*scancode*/,
                                           int act, int /*mods*/)
{
  if (act == GLFW_PRESS && key == GLFW_KEY_BACKSPACE)
  {
    mj_resetData(mj_model_, mj_data_);
    mj_forward(mj_model_, mj_data_);
  }
  if (act == GLFW_PRESS && key == GLFW_KEY_C)
  {
    show_collision_ = !show_collision_;
    mjv_opt_.geomgroup[3] = show_collision_ ? 1 : 0;
    mjv_opt_.geomgroup[2] = show_collision_ ? 0 : 1;
  }
}

void MujocoRendering::mouseButtonCallbackImpl(GLFWwindow* window, int button, int act, int /*mods*/)
{
  button_left_ = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS);
  button_middle_ = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS);
  button_right_ = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS);

  glfwGetCursorPos(window, &lastx_, &lasty_);

  (void)button;
  (void)act;
}

void MujocoRendering::mouseMoveCallbackImpl(GLFWwindow* window, double xpos, double ypos)
{
  if (!button_left_ && !button_middle_ && !button_right_)
  {
    return;
  }

  double dx = xpos - lastx_;
  double dy = ypos - lasty_;
  lastx_ = xpos;
  lasty_ = ypos;

  int width, height;
  glfwGetWindowSize(window, &width, &height);

  bool mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
                    glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS);

  mjtMouse action;
  if (button_right_)
  {
    action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
  }
  else if (button_left_)
  {
    action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
  }
  else
  {
    action = mjMOUSE_ZOOM;
  }

  mjv_moveCamera(mj_model_, action, dx / height, dy / height, &mjv_scn_, &mjv_cam_);
}

void MujocoRendering::scrollCallbackImpl(GLFWwindow* /*window*/, double /*xoffset*/, double yoffset)
{
  mjv_moveCamera(mj_model_, mjMOUSE_ZOOM, 0, -0.05 * yoffset, &mjv_scn_, &mjv_cam_);
}
} // namespace mujoco_sim
