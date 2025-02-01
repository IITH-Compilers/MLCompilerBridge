//===- agent.cpp - RL Agent/Model for ONNX Runner --------------*- C++ -*-===//
//
// Part of the MLCompilerBridge Project, under the Apache License v2.0 with LLVM
// Exceptions. See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the Agent class, which is a wrapper around the ONNXModel
/// class.
///
//===----------------------------------------------------------------------===//

#include "MLModelRunner/ONNXModelRunner/agent.h"
#include "MLModelRunner/Utils/Debug.h"
#include <algorithm>
#include <cmath>
#include <iterator>

namespace MLBridge {

Agent::Agent(std::string modelPath) {
  // Create model object here
  this->model = new ONNXModel(modelPath.c_str());
}

unsigned Agent::computeAction(Observation &input) {
  // Call model on input
  assert(input.size() > 0);
  llvm::SmallVector<float, 100> model_input(input.begin(), input.end());
  llvm::SmallVector<float, 100> model_output;

  this->model->run(model_input, model_output);

  // select action from model output
  auto max = std::max_element(model_output.begin(),
                              model_output.end()); // [2, 4)
  int argmaxVal = std::distance(model_output.begin(), max);

  // MLBRIDGE_DEBUG(
  //     std::cout << "---------------MODEL OUTPUT VECTOR:----------------\n";
  //     for (auto e
  //          : model_output) { std::cout << e << " "; } std::cout
  //     << "\nmax value and index are " << *max << " " << argmaxVal << "\n";);
  return argmaxVal;
}

} // namespace MLBridge
