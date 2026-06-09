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
#include <unordered_map>
#include <vector>

#include <eigen3/Eigen/Dense>

template <typename T>
inline T exponential_moving_average(const T output, const T current, const double alpha)
{
  return (1.0 - alpha) * current + alpha * output;
}

template <typename FieldType>
void filter_joint_values(const std::vector<std::string>& msg_names, const FieldType& msg_values,
                         const std::vector<std::string>& desired_joint_names,
                         Eigen::VectorXd& output)
{
  std::unordered_map<std::string, size_t> name_to_index;
  for (size_t i = 0; i < desired_joint_names.size(); ++i)
  {
    name_to_index[desired_joint_names[i]] = i;
  }

  for (size_t i = 0; i < msg_names.size(); ++i)
  {
    if (msg_names[i].empty() || msg_names[i].size() > 1000)
    {
      continue;
    }
    auto it = name_to_index.find(msg_names[i]);
    if (it != name_to_index.end() && i < msg_values.size())
    {
      output(it->second) = msg_values[i];
    }
  }
}
