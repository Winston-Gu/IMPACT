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
#include <Eigen/src/Core/GlobalFunctions.h>
#include <Eigen/src/Core/Matrix.h>

inline Eigen::VectorXd get_friction(const Eigen::VectorXd& dq, Eigen::VectorXd fp1,
                                    Eigen::VectorXd fp2, Eigen::VectorXd fp3)
{
  return (fp1.array() / (Eigen::VectorXd::Ones(dq.size()).array() +
                         (-fp2.array() * (dq.array() + fp3.array())).exp()) -
          fp1.array() /
              (Eigen::VectorXd::Ones(dq.size()).array() + (-fp2.array() * fp3.array()).exp()))
      .matrix();
}
