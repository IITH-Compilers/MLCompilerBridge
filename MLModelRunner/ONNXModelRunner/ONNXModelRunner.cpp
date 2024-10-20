//===- ONNXModelRunner.cpp - ONNX Runner ------------------------*- C++ -*-===//
//
// Part of the MLCompilerBridge Project, under the Apache License v2.0 with LLVM
// Exceptions. See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the ONNXModelRunner class to support ML model inference
/// via ONNX.
///
//===----------------------------------------------------------------------===//

#include "MLModelRunner/ONNXModelRunner/ONNXModelRunner.h"
#include "SerDes/baseSerDes.h"

using namespace llvm;
namespace MLBridge {

ONNXModelRunner::ONNXModelRunner(Environment *env,
                                 std::map<std::string, Agent *> agents,
                                 LLVMContext *Ctx)
    : MLModelRunner(Kind::ONNX, Ctx), env(env), agents(agents) {}

void ONNXModelRunner::addAgent(Agent *agent, std::string name) {
  if (agents.find(name) == agents.end()) {
    agents[name] = agent;
  } else {
    // throw error
    std::cerr << "ERROR: Agent with the name " << name
              << " already exists. Please give a different name!\n";
    exit(1);
  }
}

void ONNXModelRunner::computeAction(Observation &obs) {
  while (true) {
    Action action;
    // current agent
    auto current_agent = this->agents[this->env->getNextAgent()];
    action = current_agent->computeAction(obs);
    this->env->step(action);
    if (this->env->checkDone()) {
      std::cout << "Done🎉\n";
      break;
    }
  }
}

void *ONNXModelRunner::evaluateUntyped() {
  Observation &obs = env->reset();
  computeAction(obs);
  return new int(0);
}

} // namespace MLBridge
