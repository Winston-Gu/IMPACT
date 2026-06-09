#include "mujoco_sim/system_interface.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <future>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <limits>
#include <memory>
#include <mutex>
#include <rclcpp/clock.hpp>
#include <rclcpp/executors/single_threaded_executor.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp/node.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/float64.hpp>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <GLFW/glfw3.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "hardware_interface/handle.hpp"
#include "hardware_interface/system_interface.hpp"
#include "hardware_interface/types/hardware_interface_return_values.hpp"
#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "mujoco_sim/mujoco_simulator.h"
#include "rclcpp/rclcpp.hpp"

namespace mujoco_sim
{

namespace
{
std::vector<std::string> split_csv(const std::string& value)
{
  std::vector<std::string> out;
  std::string current;
  for (char ch : value)
  {
    if (ch == ',')
    {
      if (!current.empty())
      {
        out.push_back(current);
        current.clear();
      }
      continue;
    }
    if (ch != ' ' && ch != '\t' && ch != '\n' && ch != '\r')
    {
      current.push_back(ch);
    }
  }
  if (!current.empty())
  {
    out.push_back(current);
  }
  return out;
}

bool parse_bool(const std::string& value, bool fallback)
{
  if (value.empty())
  {
    return fallback;
  }
  std::string lowered = value;
  std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (lowered == "1" || lowered == "true" || lowered == "yes" || lowered == "on")
  {
    return true;
  }
  if (lowered == "0" || lowered == "false" || lowered == "no" || lowered == "off")
  {
    return false;
  }
  return fallback;
}

std::string normalize_namespace(const std::string& value)
{
  if (value.empty())
  {
    return "";
  }
  std::string trimmed = value;
  if (trimmed.front() == '/')
  {
    trimmed.erase(trimmed.begin());
  }
  while (!trimmed.empty() && trimmed.back() == '/')
  {
    trimmed.pop_back();
  }
  return trimmed;
}

} // namespace

struct CameraStream
{
  std::string name;
  std::string frame_id;
  int camera_id = -1;
  sensor_msgs::msg::CameraInfo info_msg;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub;
  rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr compressed_pub;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr info_pub;
};

class MujocoCameraPublisher
{
public:
  MujocoCameraPublisher(const rclcpp::Node::SharedPtr& node,
                        const std::vector<std::string>& camera_names,
                        const std::vector<std::string>& frame_ids,
                        const std::string& camera_namespace, int width, int height,
                        double publish_rate, bool publish_compressed, int jpeg_quality)
      : node_(node), camera_namespace_(normalize_namespace(camera_namespace)), width_(width),
        height_(height), publish_compressed_(publish_compressed), jpeg_quality_(jpeg_quality)
  {
    if (publish_rate <= 0.0)
    {
      throw std::runtime_error("camera publish_rate must be > 0");
    }

    const size_t count = camera_names.size();
    streams_.reserve(count);
    for (size_t i = 0; i < count; ++i)
    {
      CameraStream stream;
      stream.name = camera_names[i];
      if (i < frame_ids.size() && !frame_ids[i].empty())
      {
        stream.frame_id = frame_ids[i];
      }
      else
      {
        std::string base = stream.name;
        const std::string suffix = "_camera";
        if (base.size() > suffix.size() &&
            base.compare(base.size() - suffix.size(), suffix.size(), suffix) == 0)
        {
          base = base.substr(0, base.size() - suffix.size());
        }
        stream.frame_id = base + "_color_optical_frame";
      }

      const std::string prefix = camera_namespace_.empty() ? "" : "/" + camera_namespace_;
      const std::string base_topic = prefix + "/" + stream.name + "/color/image_rect_raw";
      const std::string info_topic = prefix + "/" + stream.name + "/color/camera_info";

      stream.image_pub = node_->create_publisher<sensor_msgs::msg::Image>(base_topic, 10);
      stream.info_pub = node_->create_publisher<sensor_msgs::msg::CameraInfo>(info_topic, 10);
      if (publish_compressed_)
      {
        stream.compressed_pub = node_->create_publisher<sensor_msgs::msg::CompressedImage>(
            base_topic + "/compressed", 10);
      }

      streams_.push_back(stream);
    }

    rgb_.resize(static_cast<size_t>(width_) * static_cast<size_t>(height_) * 3);
    bgr_.resize(static_cast<size_t>(width_) * static_cast<size_t>(height_) * 3);

    const auto period = std::chrono::duration<double>(1.0 / publish_rate);
    timer_ = node_->create_wall_timer(std::chrono::duration_cast<std::chrono::nanoseconds>(period),
                                      [this]() { onTimer(); });
  }

  ~MujocoCameraPublisher()
  {
    if (initialized_)
    {
      mjr_freeContext(&context_);
      mjv_freeScene(&scene_);
    }
    if (window_ != nullptr)
    {
      glfwDestroyWindow(window_);
      window_ = nullptr;
    }
  }

private:
  void onTimer()
  {
    auto& sim = MuJoCoSimulator::getInstance();
    if (sim.m == nullptr || sim.d == nullptr)
    {
      return;
    }
    if (!ensureInitialized(sim.m))
    {
      return;
    }

    std::lock_guard<std::mutex> lock(sim.state_mutex);
    glfwMakeContextCurrent(window_);
    mjr_setBuffer(mjFB_OFFSCREEN, &context_);

    for (auto& stream : streams_)
    {
      if (stream.camera_id < 0)
      {
        continue;
      }
      cam_.type = mjCAMERA_FIXED;
      cam_.fixedcamid = stream.camera_id;
      mjv_updateScene(sim.m, sim.d, &opt_, nullptr, &cam_, mjCAT_ALL, &scene_);
      mjr_render(viewport_, &scene_, &context_);
      mjr_readPixels(rgb_.data(), nullptr, viewport_, &context_);

      for (int y = 0; y < height_; ++y)
      {
        const size_t src_row = static_cast<size_t>(height_ - 1 - y) * width_ * 3;
        const size_t dst_row = static_cast<size_t>(y) * width_ * 3;
        for (int x = 0; x < width_; ++x)
        {
          const size_t src = src_row + static_cast<size_t>(x) * 3;
          const size_t dst = dst_row + static_cast<size_t>(x) * 3;
          bgr_[dst + 0] = rgb_[src + 2];
          bgr_[dst + 1] = rgb_[src + 1];
          bgr_[dst + 2] = rgb_[src + 0];
        }
      }

      const auto stamp = node_->now();
      sensor_msgs::msg::Image image_msg;
      image_msg.header.stamp = stamp;
      image_msg.header.frame_id = stream.frame_id;
      image_msg.height = static_cast<uint32_t>(height_);
      image_msg.width = static_cast<uint32_t>(width_);
      image_msg.encoding = "bgr8";
      image_msg.is_bigendian = false;
      image_msg.step = static_cast<uint32_t>(width_ * 3);
      image_msg.data = bgr_;
      stream.image_pub->publish(image_msg);

      auto info_msg = stream.info_msg;
      info_msg.header.stamp = stamp;
      info_msg.header.frame_id = stream.frame_id;
      stream.info_pub->publish(info_msg);

      if (publish_compressed_ && stream.compressed_pub != nullptr)
      {
        cv::Mat image(height_, width_, CV_8UC3, bgr_.data());
        std::vector<unsigned char> encoded;
        const std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, jpeg_quality_};
        if (cv::imencode(".jpg", image, encoded, params))
        {
          sensor_msgs::msg::CompressedImage compressed_msg;
          compressed_msg.header.stamp = stamp;
          compressed_msg.header.frame_id = stream.frame_id;
          compressed_msg.format = "jpeg";
          compressed_msg.data = std::move(encoded);
          stream.compressed_pub->publish(compressed_msg);
        }
      }
    }
  }

  bool ensureInitialized(mjModel* model)
  {
    if (initialized_ || failed_init_)
    {
      return initialized_;
    }
    if (!glfwInit())
    {
      RCLCPP_ERROR(node_->get_logger(), "Failed to initialize GLFW for MuJoCo cameras.");
      failed_init_ = true;
      return false;
    }

    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    glfwWindowHint(GLFW_DOUBLEBUFFER, GLFW_FALSE);
    window_ = glfwCreateWindow(width_, height_, "MuJoCo Offscreen", nullptr, nullptr);
    if (window_ == nullptr)
    {
      RCLCPP_ERROR(node_->get_logger(), "Failed to create GLFW window for MuJoCo cameras.");
      failed_init_ = true;
      return false;
    }
    glfwMakeContextCurrent(window_);

    mjv_defaultCamera(&cam_);
    mjv_defaultOption(&opt_);
    mjv_defaultScene(&scene_);
    mjr_defaultContext(&context_);
    mjv_makeScene(model, &scene_, 2000);
    mjr_makeContext(model, &context_, mjFONTSCALE_150);

    viewport_ = {0, 0, width_, height_};

    for (auto& stream : streams_)
    {
      stream.camera_id = mj_name2id(model, mjOBJ_CAMERA, stream.name.c_str());
      if (stream.camera_id < 0)
      {
        RCLCPP_WARN(node_->get_logger(), "MuJoCo camera '%s' not found.", stream.name.c_str());
        continue;
      }
      fillCameraInfo(stream, model, stream.camera_id);
    }

    initialized_ = true;
    return true;
  }

  void fillCameraInfo(CameraStream& stream, const mjModel* model, int camera_id)
  {
    const double fovy_deg = model->cam_fovy[camera_id];
    const double fovy = fovy_deg * M_PI / 180.0;
    const double fy = 0.5 * static_cast<double>(height_) / std::tan(fovy * 0.5);
    const double fx = fy;
    const double cx = 0.5 * static_cast<double>(width_);
    const double cy = 0.5 * static_cast<double>(height_);

    stream.info_msg.width = static_cast<uint32_t>(width_);
    stream.info_msg.height = static_cast<uint32_t>(height_);
    stream.info_msg.distortion_model = "plumb_bob";
    stream.info_msg.d.assign(5, 0.0);
    stream.info_msg.k = {fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0};
    stream.info_msg.r = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    stream.info_msg.p = {fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0};
  }

  rclcpp::Node::SharedPtr node_;
  std::vector<CameraStream> streams_;
  std::string camera_namespace_;
  int width_;
  int height_;
  bool publish_compressed_;
  int jpeg_quality_;
  rclcpp::TimerBase::SharedPtr timer_;
  bool initialized_ = false;
  bool failed_init_ = false;
  GLFWwindow* window_ = nullptr;
  mjvScene scene_;
  mjvOption opt_;
  mjvCamera cam_;
  mjrContext context_;
  mjrRect viewport_{0, 0, 0, 0};
  std::vector<unsigned char> rgb_;
  std::vector<unsigned char> bgr_;
};

Simulator::CallbackReturn Simulator::on_init(const hardware_interface::HardwareInfo& info)
{
  // Keep an internal copy of the given configuration
  if (hardware_interface::SystemInterface::on_init(info) != Simulator::CallbackReturn::SUCCESS)
  {
    return Simulator::CallbackReturn::ERROR;
  }

  info_ = info;

  if (info_.joints.size() == 0)
  {
    RCLCPP_ERROR(rclcpp::get_logger("Simulator"), "No joints found");
    return CallbackReturn::ERROR;
  }

  // Get the clock
  clock = std::make_shared<rclcpp::Clock>(RCL_SYSTEM_TIME);

  auto arm_id_it = info_.hardware_parameters.find("arm_id");
  auto prefix_it = info_.hardware_parameters.find("prefix");
  std::string arm_id = arm_id_it != info_.hardware_parameters.end() ? arm_id_it->second : "fr3";
  std::string prefix = prefix_it != info_.hardware_parameters.end() ? prefix_it->second : "";
  std::string joint_prefix = prefix.empty() ? "" : prefix + "_";
  object_frame_id_ = prefix.empty() ? arm_id + "_link0" : prefix + "_" + arm_id + "_link0";
  MuJoCoSimulator::getInstance().setGripperJointNames(joint_prefix + arm_id + "_finger_joint1",
                                                      joint_prefix + arm_id + "_finger_joint2");
  auto gripper_kp_it = info_.hardware_parameters.find("gripper_kp");
  auto gripper_kd_it = info_.hardware_parameters.find("gripper_kd");
  auto gripper_force_it = info_.hardware_parameters.find("gripper_max_force");
  auto cube_body_it = info_.hardware_parameters.find("cube_body_name");
  gripper_kp_ =
      gripper_kp_it != info_.hardware_parameters.end() ? std::stod(gripper_kp_it->second) : 2000.0;
  gripper_kd_ =
      gripper_kd_it != info_.hardware_parameters.end() ? std::stod(gripper_kd_it->second) : 5.0;
  cube_body_name_ =
      cube_body_it != info_.hardware_parameters.end() ? cube_body_it->second : "pick_cube";
  double gripper_force = gripper_force_it != info_.hardware_parameters.end()
                             ? std::stod(gripper_force_it->second)
                             : 70.0;
  MuJoCoSimulator::getInstance().setGripperGains(gripper_kp_, gripper_kd_, gripper_force);

  // Start the simulator in parallel.
  // Let the thread's destructor clean-up all resources
  // once users close the simulation window.
  m_mujoco_model = info_.hardware_parameters["mujoco_model"];
  m_simulation = std::thread(MuJoCoSimulator::simulate, m_mujoco_model);
  m_simulation.detach();

  gripper_node_ = std::make_shared<rclcpp::Node>("mujoco_gripper_target");
  std::string gripper_target_topic = "/mujoco_gripper/target_width";
  auto topic_it = info_.hardware_parameters.find("gripper_target_topic");
  if (topic_it != info_.hardware_parameters.end())
  {
    gripper_target_topic = topic_it->second;
  }
  gripper_sub_ = gripper_node_->create_subscription<std_msgs::msg::Float64>(
      gripper_target_topic, 10, [](const std_msgs::msg::Float64::SharedPtr msg)
      { MuJoCoSimulator::getInstance().setGripperTargetWidth(msg->data); });
  std::string gripper_force_topic = "/mujoco_gripper/max_force";
  auto force_it = info_.hardware_parameters.find("gripper_force_topic");
  if (force_it != info_.hardware_parameters.end())
  {
    gripper_force_topic = force_it->second;
  }
  gripper_force_sub_ = gripper_node_->create_subscription<std_msgs::msg::Float64>(
      gripper_force_topic, 10, [this](const std_msgs::msg::Float64::SharedPtr msg)
      { MuJoCoSimulator::getInstance().setGripperGains(gripper_kp_, gripper_kd_, msg->data); });
  gripper_executor_.add_node(gripper_node_);
  {
    auto names_it = info_.hardware_parameters.find("camera_names");
    std::vector<std::string> camera_names = names_it != info_.hardware_parameters.end()
                                                ? split_csv(names_it->second)
                                                : std::vector<std::string>{};
    if (camera_names.empty())
    {
      camera_names = {"front_camera", "side_camera"};
    }

    auto frames_it = info_.hardware_parameters.find("camera_frame_ids");
    std::vector<std::string> frame_ids = frames_it != info_.hardware_parameters.end()
                                             ? split_csv(frames_it->second)
                                             : std::vector<std::string>{};

    std::string camera_namespace = "camera";
    auto ns_it = info_.hardware_parameters.find("camera_namespace");
    if (ns_it != info_.hardware_parameters.end())
    {
      camera_namespace = ns_it->second;
    }

    int camera_width = 640;
    int camera_height = 480;
    double camera_rate = 30.0;
    int jpeg_quality = 90;
    bool publish_compressed = true;

    auto width_it = info_.hardware_parameters.find("camera_width");
    if (width_it != info_.hardware_parameters.end())
    {
      camera_width = std::stoi(width_it->second);
    }
    auto height_it = info_.hardware_parameters.find("camera_height");
    if (height_it != info_.hardware_parameters.end())
    {
      camera_height = std::stoi(height_it->second);
    }
    auto rate_it = info_.hardware_parameters.find("camera_rate");
    if (rate_it != info_.hardware_parameters.end())
    {
      camera_rate = std::stod(rate_it->second);
    }
    auto quality_it = info_.hardware_parameters.find("camera_jpeg_quality");
    if (quality_it != info_.hardware_parameters.end())
    {
      jpeg_quality = std::stoi(quality_it->second);
    }
    auto compressed_it = info_.hardware_parameters.find("camera_publish_compressed");
    if (compressed_it != info_.hardware_parameters.end())
    {
      publish_compressed = parse_bool(compressed_it->second, publish_compressed);
    }

    if (!camera_names.empty())
    {
      camera_publisher_ = std::make_shared<MujocoCameraPublisher>(
          gripper_node_, camera_names, frame_ids, camera_namespace, camera_width, camera_height,
          camera_rate, publish_compressed, jpeg_quality);
    }
  }
  gripper_spin_active_.store(true);
  gripper_spin_thread_ = std::thread(
      [this]()
      {
        while (gripper_spin_active_.load())
        {
          gripper_executor_.spin_some();
          std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
      });

  reset_scene_srv_ = gripper_node_->create_service<std_srvs::srv::Trigger>(
      "/mujoco_sim/reset_scene",
      [this](const std::shared_ptr<std_srvs::srv::Trigger::Request> /*request*/,
             std::shared_ptr<std_srvs::srv::Trigger::Response> response)
      {
        bool ok = MuJoCoSimulator::getInstance().requestSceneReset(5.0);
        response->success = ok;
        response->message = ok ? "scene reset" : "scene reset failed";
      });
  object_mass_srv_ = gripper_node_->create_service<std_srvs::srv::Trigger>(
      "/mujoco_sim/get_object_mass",
      [this](const std::shared_ptr<std_srvs::srv::Trigger::Request> /*request*/,
             std::shared_ptr<std_srvs::srv::Trigger::Response> response)
      {
        double mass = 0.0;
        bool ok = MuJoCoSimulator::getInstance().getBodyMass(cube_body_name_, mass);
        response->success = ok;
        if (ok)
        {
          response->message = std::to_string(mass);
        }
        else
        {
          response->message = "body not found";
        }
      });

  cube_pose_pub_ = gripper_node_->create_publisher<geometry_msgs::msg::PoseStamped>(
      "/mujoco_objects/pick_cube_pose", 10);
  basket_pose_pub_ = gripper_node_->create_publisher<geometry_msgs::msg::PoseStamped>(
      "/mujoco_objects/basket_pose", 10);
  object_pose_timer_ = gripper_node_->create_wall_timer(
      std::chrono::milliseconds(20),
      [this]()
      {
        auto quat_conjugate = [](const std::array<double, 4>& q)
        { return std::array<double, 4>{{q[0], -q[1], -q[2], -q[3]}}; };
        auto quat_multiply = [](const std::array<double, 4>& a, const std::array<double, 4>& b)
        {
          return std::array<double, 4>{{a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
                                        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
                                        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
                                        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]}};
        };
        auto quat_rotate = [&](const std::array<double, 4>& q, const std::array<double, 3>& v)
        {
          const std::array<double, 4> vq{{0.0, v[0], v[1], v[2]}};
          const auto qi = quat_conjugate(q);
          const auto tmp = quat_multiply(q, vq);
          const auto out = quat_multiply(tmp, qi);
          return std::array<double, 3>{{out[1], out[2], out[3]}};
        };

        std::array<double, 3> cube_pos;
        std::array<double, 4> cube_quat;
        std::array<double, 3> basket_pos;
        std::array<double, 4> basket_quat;
        std::array<double, 3> base_pos;
        std::array<double, 4> base_quat;

        bool has_cube =
            MuJoCoSimulator::getInstance().getBodyPose("pick_cube", cube_pos, cube_quat);
        bool has_basket =
            MuJoCoSimulator::getInstance().getBodyPose("target_basket", basket_pos, basket_quat);
        bool has_base =
            MuJoCoSimulator::getInstance().getBodyPose("fr3_link0", base_pos, base_quat);

        if (!has_cube && !has_basket)
        {
          return;
        }

        if (has_base)
        {
          const auto base_inv = quat_conjugate(base_quat);
          if (has_cube)
          {
            const std::array<double, 3> rel{
                {cube_pos[0] - base_pos[0], cube_pos[1] - base_pos[1], cube_pos[2] - base_pos[2]}};
            cube_pos = quat_rotate(base_inv, rel);
            cube_quat = quat_multiply(base_inv, cube_quat);
          }
          if (has_basket)
          {
            const std::array<double, 3> rel{{basket_pos[0] - base_pos[0],
                                             basket_pos[1] - base_pos[1],
                                             basket_pos[2] - base_pos[2]}};
            basket_pos = quat_rotate(base_inv, rel);
            basket_quat = quat_multiply(base_inv, basket_quat);
          }
        }

        const auto stamp = gripper_node_->now();
        if (has_cube)
        {
          geometry_msgs::msg::PoseStamped msg;
          msg.header.stamp = stamp;
          msg.header.frame_id = object_frame_id_;
          msg.pose.position.x = cube_pos[0];
          msg.pose.position.y = cube_pos[1];
          msg.pose.position.z = cube_pos[2];
          msg.pose.orientation.w = cube_quat[0];
          msg.pose.orientation.x = cube_quat[1];
          msg.pose.orientation.y = cube_quat[2];
          msg.pose.orientation.z = cube_quat[3];
          cube_pose_pub_->publish(msg);
        }

        if (has_basket)
        {
          geometry_msgs::msg::PoseStamped msg;
          msg.header.stamp = stamp;
          msg.header.frame_id = object_frame_id_;
          msg.pose.position.x = basket_pos[0];
          msg.pose.position.y = basket_pos[1];
          msg.pose.position.z = basket_pos[2];
          msg.pose.orientation.w = basket_quat[0];
          msg.pose.orientation.x = basket_quat[1];
          msg.pose.orientation.y = basket_quat[2];
          msg.pose.orientation.z = basket_quat[3];
          basket_pose_pub_->publish(msg);
        }
      });

  m_positions.resize(info_.joints.size(), 0.0);
  m_velocities.resize(info_.joints.size(), 0.0);
  m_efforts.resize(info_.joints.size(), 0.0);

  m_effort_commands.resize(info_.joints.size(), 0.0);

  int joint_idx = 0;
  for (const hardware_interface::ComponentInfo& joint : info_.joints)
  {
    RCLCPP_DEBUG_STREAM(rclcpp::get_logger("Simulator"), "Found joint: " << joint.name);
    if (joint.command_interfaces.size() != 1)
    {
      RCLCPP_ERROR(rclcpp::get_logger("Simulator"), "Joint '%s' needs 1 command interface.",
                   joint.name.c_str());
      return Simulator::CallbackReturn::ERROR;
    }
    if (joint.command_interfaces[0].name != hardware_interface::HW_IF_EFFORT)
    {
      RCLCPP_ERROR(rclcpp::get_logger("Simulator"),
                   "Joint '%s' needs the following command interface: %s.", joint.name.c_str(),
                   hardware_interface::HW_IF_EFFORT);
      return Simulator::CallbackReturn::ERROR;
    }
    if (joint.state_interfaces.size() != 3)
    {
      RCLCPP_ERROR(rclcpp::get_logger("Simulator"), "Joint '%s' needs 3 state interfaces.",
                   joint.name.c_str());

      return Simulator::CallbackReturn::ERROR;
    }

    if (!(joint.state_interfaces[0].name == hardware_interface::HW_IF_POSITION ||
          joint.state_interfaces[0].name == hardware_interface::HW_IF_VELOCITY ||
          joint.state_interfaces[0].name == hardware_interface::HW_IF_EFFORT))
    {
      RCLCPP_ERROR(rclcpp::get_logger("Simulator"),
                   "Joint '%s' needs the following state interfaces in that "
                   "order: %s, %s, and %s.",
                   joint.name.c_str(), hardware_interface::HW_IF_POSITION,
                   hardware_interface::HW_IF_VELOCITY, hardware_interface::HW_IF_EFFORT);

      return Simulator::CallbackReturn::ERROR;
    }

    if (!(joint.state_interfaces[0].initial_value.empty()))
    {
      m_positions[joint_idx] = joint.state_interfaces[0].initial_value.at(0);
    }
    joint_idx++;
  }

  return Simulator::CallbackReturn::SUCCESS;
}

std::vector<hardware_interface::StateInterface> Simulator::export_state_interfaces()
{
  std::vector<hardware_interface::StateInterface> state_interfaces;
  for (std::size_t i = 0; i < info_.joints.size(); i++)
  {
    state_interfaces.emplace_back(hardware_interface::StateInterface(
        info_.joints[i].name, hardware_interface::HW_IF_POSITION, &m_positions[i]));
    state_interfaces.emplace_back(hardware_interface::StateInterface(
        info_.joints[i].name, hardware_interface::HW_IF_VELOCITY, &m_velocities[i]));
    state_interfaces.emplace_back(hardware_interface::StateInterface(
        info_.joints[i].name, hardware_interface::HW_IF_EFFORT, &m_efforts[i]));
  }

  return state_interfaces;
}

std::vector<hardware_interface::CommandInterface> Simulator::export_command_interfaces()
{
  std::vector<hardware_interface::CommandInterface> command_interfaces;
  for (std::size_t i = 0; i < info_.joints.size(); i++)
  {
    command_interfaces.emplace_back(hardware_interface::CommandInterface(
        info_.joints[i].name, hardware_interface::HW_IF_EFFORT, &m_effort_commands[i]));
  }

  return command_interfaces;
}

Simulator::return_type Simulator::prepare_command_mode_switch(
    [[maybe_unused]] const std::vector<std::string>& start_interfaces,
    [[maybe_unused]] const std::vector<std::string>& stop_interfaces)
{
  return return_type::OK;
}

Simulator::return_type Simulator::read([[maybe_unused]] const rclcpp::Time& time,
                                       [[maybe_unused]] const rclcpp::Duration& period)
{
  /*RCLCPP_INFO_STREAM_THROTTLE(rclcpp::get_logger("Simulator"), *clock, 1000,
   * "Reading state values" << m_positions[0]);*/
  MuJoCoSimulator::getInstance().read(m_positions, m_velocities, m_efforts);
  return return_type::OK;
}

Simulator::return_type Simulator::write([[maybe_unused]] const rclcpp::Time& time,
                                        [[maybe_unused]] const rclcpp::Duration& period)
{
  /*RCLCPP_INFO_STREAM_THROTTLE(rclcpp::get_logger("Simulator"), *clock, 1000,
   * "Writing effort commands" << m_effort_commands[0]);*/
  MuJoCoSimulator::getInstance().write(m_effort_commands);
  return return_type::OK;
}

} // namespace mujoco_sim

#include "pluginlib/class_list_macros.hpp"

PLUGINLIB_EXPORT_CLASS(mujoco_sim::Simulator, hardware_interface::SystemInterface)
