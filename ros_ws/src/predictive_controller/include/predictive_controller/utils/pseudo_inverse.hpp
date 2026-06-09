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
#include <Eigen/LU>
#include <Eigen/SVD>

namespace predictive_controller
{

inline Eigen::MatrixXd pseudo_inverse(const Eigen::MatrixXd& matrix, double epsilon = 1e-4)
{
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::VectorXd singularValues = svd.singularValues();
  Eigen::MatrixXd S = Eigen::MatrixXd::Zero(svd.matrixV().cols(), svd.matrixU().rows());
  for (int i = 0; i < singularValues.size(); ++i)
  {
    S(i, i) = singularValues[i] / (singularValues[i] * singularValues[i] + epsilon * epsilon);
  }

  return svd.matrixV() * S * svd.matrixU().transpose();
}

inline Eigen::MatrixXd pseudo_inverse_moore_penrose(const Eigen::MatrixXd& matrix,
                                                    double epsilon = 1e-6)
{
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::VectorXd singularValues = svd.singularValues();
  Eigen::MatrixXd S = Eigen::MatrixXd::Zero(svd.matrixV().cols(), svd.matrixU().rows());
  for (int i = 0; i < singularValues.size(); ++i)
  {
    S(i, i) = singularValues[i] > epsilon ? 1.0 / singularValues[i] : 0.0;
  }

  return svd.matrixV() * S * svd.matrixU().transpose();
}

inline bool is_near_singular(const Eigen::MatrixXd& matrix, double epsilon = 1e-6)
{
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::VectorXd singularValues = svd.singularValues();
  return (singularValues.array() < epsilon).any();
}

} // namespace predictive_controller
