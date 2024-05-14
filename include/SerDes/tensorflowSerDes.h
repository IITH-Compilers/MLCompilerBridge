//=== SerDes/tensorflowSerDes.h - SerDes for TF support ---*- C++ ---------===//
//
// Part of the MLCompilerBridge Project, under the Apache License v2.0 with LLVM
// Exceptions. See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Serialization/Deserialization to support TF AOT models.
///
//===----------------------------------------------------------------------===//

#ifndef TENSORFLOW_SERIALIZER_H
#define TENSORFLOW_SERIALIZER_H

#include "SerDes/baseSerDes.h"
#include "tensorflow/compiler/tf2xla/xla_compiled_cpu_function.h"

namespace MLBridge {
/// TensorflowSerDes - Serialization/Deserialization to support TF AOT models.
class TensorflowSerDes : public BaseSerDes {
public:
  TensorflowSerDes() : BaseSerDes(SerDesKind::Tensorflow) {}

  static bool classof(const BaseSerDes *S) {
    return S->getKind() == SerDesKind::Tensorflow;
  }

#define SET_FEATURE(TYPE, _)                                                   \
  void setFeature(const std::string &, const TYPE) override;                   \
  void setFeature(const std::string &, const std::vector<TYPE> &) override;
  SUPPORTED_TYPES(SET_FEATURE)
#undef SET_FEATURE

  void setRequest(void *request) override {
    CompiledModel =
        reinterpret_cast<tensorflow::XlaCompiledCpuFunction *>(request);
  }

  void *getSerializedData() override { return nullptr; };
  void cleanDataStructures() override{};

private:
  void *deserializeUntyped(void *data) override { return nullptr; };
  tensorflow::XlaCompiledCpuFunction *CompiledModel;
};
} // namespace MLBridge

#endif
