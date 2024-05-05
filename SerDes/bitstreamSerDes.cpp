//===- bitstreamSerDes.cpp - Serializer for Bitstream -----------*- C++ -*-===//
//
// Part of the MLCompilerBridge Project, under the Apache License v2.0 with LLVM
// Exceptions. See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the BitstreamSerDes class, which is a serializer for
/// Bitstream format.
///
//===----------------------------------------------------------------------===//

#include "SerDes/bitstreamSerDes.h"
#include "MLModelRunner/Utils/Debug.h"
#include "llvm/Support/JSON.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <vector>

#define DEBUG_TYPE "bitstream-serdes"

namespace MLBridge {
void BitstreamSerDes::setFeature(const std::string &name, const int value) {
  auto *valuePtr = new int(value);
  featuresint[name] = valuePtr;
  tensorSpecs.push_back(TensorSpec::createSpec<int>(name, {1}));
  rawData.push_back(valuePtr);
}

void BitstreamSerDes::setFeature(const std::string &name, const long value) {
  auto *valuePtr = new long(value);
  featureslong[name] = valuePtr;
  tensorSpecs.push_back(TensorSpec::createSpec<long>(name, {1}));
  rawData.push_back(valuePtr);
}

void BitstreamSerDes::setFeature(const std::string &name, const float value) {
  auto *valuePtr = new float(value);
  featuresfloat[name] = valuePtr;
  tensorSpecs.push_back(TensorSpec::createSpec<float>(name, {1}));
  rawData.push_back(valuePtr);
}

void BitstreamSerDes::setFeature(const std::string &name, const double value) {
  auto *valuePtr = new double(value);
  featuresdouble[name] = valuePtr;
  tensorSpecs.push_back(TensorSpec::createSpec<double>(name, {1}));
  rawData.push_back(valuePtr);
}

void BitstreamSerDes::setFeature(const std::string &name,
                                 const std::string value) {
  auto *valuePtr = new std::string(value);
  featuresstring[name] = valuePtr;
  long size = value.length();
  tensorSpecs.push_back(TensorSpec::createSpec<uint8_t>(name, {size}));
  rawData.push_back((void *)valuePtr->c_str());
}

void BitstreamSerDes::setFeature(const std::string &name, const bool value) {
  auto *valuePtr = new bool(value);
  featuresbool[name] = valuePtr;
  tensorSpecs.push_back(TensorSpec::createSpec<uint8_t>(name, {1}));
  rawData.push_back(valuePtr);
}

void BitstreamSerDes::setFeature(const std::string &name,
                                 const std::vector<int> &value) {
  auto *valuePtr = new std::vector<int>(value);
  featuresVectorint[name] = valuePtr;
  tensorSpecs.push_back(
      TensorSpec::createSpec<int>(name, {static_cast<long>(valuePtr->size())}));
  rawData.push_back(valuePtr->data());
}

void BitstreamSerDes::setFeature(const std::string &name,
                                 const std::vector<long> &value) {
  auto *valuePtr = new std::vector<long>(value);
  featuresVectorlong[name] = valuePtr;
  tensorSpecs.push_back(TensorSpec::createSpec<long>(
      name, {static_cast<long>(valuePtr->size())}));
  rawData.push_back(valuePtr->data());
}

void BitstreamSerDes::setFeature(const std::string &name,
                                 const std::vector<float> &value) {
  auto *valuePtr = new std::vector<float>(value);
  featuresVectorfloat[name] = valuePtr;
  tensorSpecs.push_back(TensorSpec::createSpec<float>(
      name, {static_cast<long>(valuePtr->size())}));
  rawData.push_back(valuePtr->data());
}

void BitstreamSerDes::setFeature(const std::string &name,
                                 const std::vector<double> &value) {
  auto *valuePtr = new std::vector<double>(value);
  featuresVectordouble[name] = valuePtr;
  tensorSpecs.push_back(TensorSpec::createSpec<double>(
      name, {static_cast<long>(valuePtr->size())}));
  rawData.push_back(valuePtr->data());
}

void BitstreamSerDes::setFeature(const std::string &name,
                                 const std::vector<std::string> &value) {
  llvm_unreachable("Currently std::vector<std::string> not supported");
}

void BitstreamSerDes::setFeature(const std::string &name,
                                 const std::vector<bool> &value) {
  llvm_unreachable("Currently std::vector<bool> not supported");
}

void *BitstreamSerDes::getSerializedData() {
  std::unique_ptr<llvm::raw_ostream> OS =
      std::make_unique<llvm::raw_string_ostream>(Buffer);
  llvm::json::OStream J(*OS);
  J.object([&]() {
    J.attributeArray("features", [&]() {
      for (size_t I = 0; I < tensorSpecs.size(); ++I) {
        tensorSpecs[I].toJSON(J);
      }
    });
  });
  J.flush();
  OS->write("\n", 1);
  MLBRIDGE_DEBUG(std::cout << "rawData.size(): " << rawData.size() << "\n");
  for (size_t I = 0; I < rawData.size(); ++I) {
    OS->write(reinterpret_cast<const char *>(rawData[I]),
              tensorSpecs[I].getTotalTensorBufferSize());
  }
  OS->write("\n", 1);
  OS->flush();
  auto *out = new std::string(Buffer);
  cleanDataStructures();
  return out;
}

void *BitstreamSerDes::deserializeUntyped(void *data) {
  // set the message length
  auto *res = reinterpret_cast<std::string *>(data);
  this->MessageLength = res->size();
  return res->data();
}
} // namespace MLBridge
