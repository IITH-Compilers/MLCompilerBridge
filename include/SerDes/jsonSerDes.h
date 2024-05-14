//=== SerDes/jsonSerDes.h -Json Serialization/Deserialization ---*- C++ ---===//
//
// Part of the MLCompilerBridge Project, under the Apache License v2.0 with LLVM
// Exceptions. See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Json Serialization/Deserialization using LLVM's json library.
///
//===----------------------------------------------------------------------===//

#ifndef JSON_SERIALIZER_H
#define JSON_SERIALIZER_H

#include "SerDes/baseSerDes.h"
#include "llvm/Support/JSON.h"
#include <string>
#include <utility>

namespace MLBridge {
/// JsonSerDes - Json Serialization/Deserialization using LLVM's json library.
class JsonSerDes : public BaseSerDes {
public:
  JsonSerDes() : BaseSerDes(SerDesKind::Json){};

  static bool classof(const BaseSerDes *S) {
    return S->getKind() == SerDesKind::Json;
  }

#define SET_FEATURE(TYPE, _)                                                   \
  void setFeature(const std::string &name, const TYPE value) override {        \
    J[name] = value;                                                           \
  }                                                                            \
  void setFeature(const std::string &name, const std::vector<TYPE> &value)     \
      override {                                                               \
    J[name] = llvm::json::Array(value);                                        \
  }
  SUPPORTED_TYPES(SET_FEATURE)
#undef SET_FEATURE

  void *getSerializedData() override;

  void cleanDataStructures() override { J = llvm::json::Object(); }

private:
  void *deserializeUntyped(void *data) override;
  void *desJson(llvm::json::Value *V);

private:
  llvm::json::Object J;
};
} // namespace MLBridge

#endif
