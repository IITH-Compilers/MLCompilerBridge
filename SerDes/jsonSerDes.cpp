//===- jsonstreamSerDes.cpp - Serializer for JSON ---------------*- C++ -*-===//
//
// Part of the MLCompilerBridge Project, under the Apache License v2.0 with LLVM
// Exceptions. See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the JsonSerDes class, which is a wrapper around the JSON
/// C++ interface to serialize and deserialize data to and from JSON.
///
//===----------------------------------------------------------------------===//

#include "SerDes/jsonSerDes.h"
#include "MLModelRunner/Utils/DataTypes.h"
#include "MLModelRunner/Utils/Debug.h"
#include "SerDes/baseSerDes.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/JSON.h"
#include <cstdint>
#include <string>

#define DEBUG_TYPE "json-serdes"

using namespace llvm;

namespace MLBridge {
void *JsonSerDes::getSerializedData() {
  auto tempJO = J;
  auto data = json::Value(std::move(tempJO));
  auto *ret = new std::string();
  llvm::raw_string_ostream OS(*ret);
  json::OStream(OS).value(data);
  cleanDataStructures();
  return ret;
}

void *JsonSerDes::deserializeUntyped(void *data) {
  MLBRIDGE_DEBUG(std::cout << "In JsonSerDes deserializeUntyped...\n");
  auto dataStr = static_cast<std::string *>(data);
  MLBRIDGE_DEBUG(std::cout << "dataStr: " << *dataStr << "\n");
  Expected<json::Value> valueOrErr = json::parse(*dataStr);
  if (!valueOrErr) {
    auto *ret = new std::string();
    llvm::raw_string_ostream SOS(*ret);
    SOS << "Error parsing JSON: " << valueOrErr.takeError() << "\n";
    std::cerr << &ret << "\n";
    exit(1);
  }
  json::Object *ret = valueOrErr->getAsObject();
  auto val = ret->get("out");
  MLBRIDGE_DEBUG(std::cout << "Got the final array...\n";
                 std::cout << "End JsonSerDes deserializeUntyped...\n");
  return desJson(val);
}

void *JsonSerDes::desJson(json::Value *V) {
  switch (V->kind()) {
  case json::Value::Kind::Null:
    return nullptr;
  case json::Value::Kind::Number: {
    if (auto x = V->getAsInteger()) {
      IntegerType *ret = new IntegerType();
      *ret = x.getValue();
      this->MessageLength = sizeof(IntegerType);
      return ret;
    } else if (auto x = V->getAsNumber()) {
      RealType *ret = new RealType();
      *ret = x.getValue();
      this->MessageLength = sizeof(RealType);
      return ret;
    } else {
      std::cerr << "Error in desJson: Number is not int, or double\n";
      exit(1);
    }
  }
  case json::Value::Kind::String: {
    std::string *ret = new std::string();
    *ret = V->getAsString()->str();
    this->MessageLength = ret->size() * sizeof(char);
    return ret->data();
  }
  case json::Value::Kind::Boolean: {
    bool *ret = new bool();
    *ret = V->getAsBoolean().getValue();
    this->MessageLength = sizeof(bool);
    return ret;
  }
  case json::Value::Kind::Array: {
    // iterate over array and find its type
    // assume all elements are of same type and return vector of that type
    // if not, return nullptr
    auto arr = V->getAsArray();

    auto it = arr->begin();
    auto first = it;
    switch (first->kind()) {
    case json::Value::Kind::Number: {
      if (auto x = first->getAsInteger()) {
        std::vector<IntegerType> *ret = new std::vector<IntegerType>();
        for (auto it : *arr) {
          ret->push_back(it.getAsInteger().getValue());
        }
        this->MessageLength = ret->size() * sizeof(IntegerType);
        return ret->data();
      } else if (auto x = first->getAsNumber()) {
        std::vector<RealType> *ret = new std::vector<RealType>();
        for (auto it : *arr) {
          ret->push_back(it.getAsNumber().getValue());
        }
        this->MessageLength = ret->size() * sizeof(RealType);
        return ret->data();
      } else {
        std::cerr << "Error in desJson: Number is not int, or double\n";
        exit(1);
      }
    }
    case json::Value::Kind::String: {
      std::vector<std::string> *ret = new std::vector<std::string>();
      for (auto it : *arr) {
        ret->push_back(it.getAsString()->str());
      }
      this->MessageLength = ret->size() * sizeof(std::string);
      return ret->data();
    }
    case json::Value::Kind::Boolean: {
      std::vector<uint8_t> *ret = new std::vector<uint8_t>();
      for (auto it : *arr) {
        ret->push_back(it.getAsBoolean().getValue());
      }
      this->MessageLength = ret->size() * sizeof(uint8_t);
      return ret->data();
    }
    default: {
      std::cerr << "Error in desJson: Array is not of supported type\n";
      exit(1);
    }
    }
  }
  }
  return nullptr;
}
} // namespace MLBridge
