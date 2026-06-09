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

#include <Eigen/Core>

template <typename T>
inline Eigen::VectorXd saturate_torque_rate(const Eigen::VectorXd& tau_d_calculated,
                                            const Eigen::VectorXd& tau_J_d, const T& delta_tau_max)
{
  Eigen::VectorXd difference = tau_d_calculated - tau_J_d;
  Eigen::VectorXd clamped_diff = difference.cwiseMin(delta_tau_max).cwiseMax(-delta_tau_max);
  return tau_J_d + clamped_diff;
}
