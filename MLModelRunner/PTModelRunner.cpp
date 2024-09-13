//=== PTModelRunner.cpp - PTModelRunner Implementation ---*- C++ -*-===//
//
// Part of the MLCompilerBridge Project
//
//===------------------===//

#include "MLModelRunner/PTModelRunner.h"
#include "SerDes/pytorchSerDes.h"
#include "llvm/Support/ErrorHandling.h"

using namespace torch::inductor;

namespace MLBridge {

template <class TGen>
void *PTModelRunner<TGen>::evaluateUntyped() {
    if (CompiledModel->inputTensors.empty()) {
        llvm::errs() << "Input vector is empty.\n";
        return nullptr;
    }

    try {
        // Run the model with the input tensors
        auto outputs = CompiledModel->CompiledModel->run(CompiledModel->inputTensors);

        // Store the outputs
        CompiledModel->outputTensors = outputs;

        // Convert to raw data format using deserializeUntyped
        void *rawData = CompiledModel->deserializeUntyped(&CompiledModel->outputTensors);

        return rawData;
    } catch (const c10::Error &e) {
        llvm::errs() << "Error during model evaluation: " << e.what() << "\n";
        return nullptr;
    }
}

} // namespace MLBridge
