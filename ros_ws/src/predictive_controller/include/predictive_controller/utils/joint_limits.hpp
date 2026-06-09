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

#include <Eigen/Dense>

inline Eigen::VectorXd get_joint_limit_torque(const Eigen::VectorXd& joint_positions,
                                              const Eigen::VectorXd& lower_limits,
                                              const Eigen::VectorXd& upper_limits,
                                              double safe_range = 0.3, double max_torque = 5.0)
{
  Eigen::VectorXd torques = Eigen::VectorXd::Zero(joint_positions.size());

  Eigen::VectorXd dist_to_lower = joint_positions - lower_limits;
  Eigen::VectorXd dist_to_upper = upper_limits - joint_positions;

  Eigen::Array<bool, Eigen::Dynamic, 1> near_lower = (dist_to_lower.array() < safe_range);
  Eigen::Array<bool, Eigen::Dynamic, 1> near_upper = (dist_to_upper.array() < safe_range);

  Eigen::ArrayXd lower_ratios =
      ((safe_range - dist_to_lower.array()) / safe_range).cwiseMax(0.0).cwiseMin(1.0);
  Eigen::ArrayXd upper_ratios =
      ((safe_range - dist_to_upper.array()) / safe_range).cwiseMax(0.0).cwiseMin(1.0);

  torques = (near_lower.select(max_torque * lower_ratios, 0.0) -
             near_upper.select(max_torque * upper_ratios, 0.0))
                .matrix();
  return torques;
}
