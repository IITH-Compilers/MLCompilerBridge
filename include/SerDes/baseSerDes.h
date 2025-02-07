//=== SerDes/baseSerDes.h - Base for serialization and deserialization C++ ===//
//
// Part of the MLCompilerBridge Project, under the Apache License v2.0 with LLVM
// Exceptions. See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Supporting new SerDes:
/// 1. Create a new class which inherits from BaseSerDes.
/// 2. Implement the setFeature(), getSerializedData(), cleanDataStructures()
/// and deserializeUntyped() methods.
/// 3. Add the new SerDes to the enum class Kind in this class.
///
//===----------------------------------------------------------------------===//

#ifndef BASE_SERDES_H
#define BASE_SERDES_H

#include "MLModelRunner/Utils/Debug.h"
#include "google/protobuf/extension_set.h"
#include "google/protobuf/message.h"
#include "llvm/Support/raw_ostream.h"
#include "SerDes/TensorSpec.h"

#include <cassert>
#include <map>
#include <string>
#include <vector>

// TYPE, NAME
#define SUPPORTED_TYPES(M)                                                     \
  M(int, int)                                                                  \
  M(long, long)                                                                \
  M(float, float)                                                              \
  M(double, double)                                                            \
  M(std::string, string)                                                       \
  M(bool, bool)

namespace MLBridge {
/// This is the base class for SerDes. It defines the interface for the
/// serialization and deserialization of the data structures used for the
/// communication by the MLModelRunner.
/// Currently, (int, float) or (long, double), char and bool are supported.
/// Vectors of these types are supported as well.
class BaseSerDes {
public:
  enum class Kind : int { Unknown, Json, Bitstream, Protobuf, Tensorflow, Pytorch };
  Kind getKind() const { return Type; }

  /// setFeature() is used to set the features of the data structure used for
  /// communication. The features are set as key-value pairs. The key is a
  /// string and the value can be any of the supported types.
#define SET_FEATURE(TYPE, _)                                                   \
  virtual void setFeature(const std::string &, const TYPE) = 0;                \
  virtual void setFeature(const std::string &, const std::vector<TYPE> &){};
  SUPPORTED_TYPES(SET_FEATURE)
#undef SET_FEATURE

  virtual void setFeature(const std::string &name,
                          const google::protobuf::Message *value){};
  virtual void
  setFeature(const std::string &name,
             const std::vector<google::protobuf::Message *> &value){};

  // a hack to set the request and response structures in protobuf serializer
  virtual void setRequest(void *Request) {
    MLBRIDGE_DEBUG(std::cout << "In BaseSerializer setRequest...\n");
  };
  virtual void setResponse(void *Response){};
  virtual void *getSerializedData() = 0;
  virtual void *deserializeUntyped(void *data) = 0;
  size_t getMessageLength() { return MessageLength; }
  virtual void *getRequest() { return nullptr; };
  virtual void *getResponse() { return nullptr; };

protected:
  BaseSerDes(Kind Type) : Type(Type) { assert(Type != Kind::Unknown); }
  virtual void cleanDataStructures() = 0;
  const Kind Type;
  void *RequestVoid;
  void *ResponseVoid;
  size_t MessageLength;
};
} // namespace MLBridge

#endif
