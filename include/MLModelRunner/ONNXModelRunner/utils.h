//=== MLModelRunner/ONNXModelRunner/utils.h - C++ ------------------------===//
//
// Part of the MLCompilerBridge Project, under the Apache License v2.0 with LLVM
// Exceptions. See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MODELRUNNER_UTILS_H
#define ONNX_MODELRUNNER_UTILS_H

#include "llvm/ADT/SmallVector.h"

namespace MLBridge {
typedef llvm::SmallVector<float, 300> Observation;

} // namespace MLBridge

#endif // ONNX_MODELRUNNER_UTILS_H
