//=== SerDes/bitstreamSerDes.h -Bitstream Serialization/Deserialization-C++===//
//
// Part of the MLCompilerBridge Project, under the Apache License v2.0 with LLVM
// Exceptions. See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Bitstream Serialization/Deserialization which sends header information
/// followed by the raw data.
///
//===----------------------------------------------------------------------===//

#ifndef BITSTREAM_SERIALIZER_H
#define BITSTREAM_SERIALIZER_H

#include "SerDes/TensorSpec.h"
#include "SerDes/baseSerDes.h"
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace MLBridge {
/// BitstreamSerDes - Bitstream Serialization/Deserialization which sends header
/// information followed by the raw data.
class BitstreamSerDes : public BaseSerDes {
public:
  BitstreamSerDes() : BaseSerDes(SerDesKind::Bitstream) {
    Buffer = "";
    tensorSpecs = std::vector<TensorSpec>();
    rawData = std::vector<const void *>();

#define TEMPORARY_STORAGE_INIT(TYPE, NAME)                                     \
  features##NAME = {};                                                         \
  featuresVector##NAME = {};
    SUPPORTED_TYPES(TEMPORARY_STORAGE_INIT)
#undef TEMPORARY_STORAGE_INIT
  };
#define SET_FEATURE(TYPE, _)                                                   \
  void setFeature(const std::string &, const TYPE) override;                   \
  void setFeature(const std::string &, const std::vector<TYPE> &) override;
  SUPPORTED_TYPES(SET_FEATURE)
#undef SET_FEATURE

  void *getSerializedData() override;

  void cleanDataStructures() override {
    Buffer = "";
    tensorSpecs = std::vector<TensorSpec>();
    rawData = std::vector<const void *>();

#define TEMPORARY_STORAGE_CLEAN(TYPE, NAME)                                    \
  for (auto &it : features##NAME) {                                            \
    delete it.second;                                                          \
  }                                                                            \
  features##NAME.clear();                                                      \
  features##NAME = {};                                                         \
  for (auto &it : featuresVector##NAME) {                                      \
    delete it.second;                                                          \
  }                                                                            \
  featuresVector##NAME.clear();                                                \
  featuresVector##NAME = {};
    SUPPORTED_TYPES(TEMPORARY_STORAGE_CLEAN)
#undef TEMPORARY_STORAGE_CLEAN
  }

private:
  void *deserializeUntyped(void *) override;
  std::vector<TensorSpec> tensorSpecs;
  std::vector<const void *> rawData;
  std::string Buffer;

#define TEMPORARY_STORAGE_DEF(TYPE, NAME)                                      \
  std::map<std::string, TYPE *> features##NAME;                                \
  std::map<std::string, std::vector<TYPE> *> featuresVector##NAME;
  SUPPORTED_TYPES(TEMPORARY_STORAGE_DEF)
#undef TEMPORARY_STORAGE_DEF
};
} // namespace MLBridge

#endif
