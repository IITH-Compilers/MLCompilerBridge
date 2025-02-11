//=== PTModelRunner.cpp - PTModelRunner Implementation ---*- C++ -*-===//
//
// Part of the MLCompilerBridge Project
//
//===------------------===//

#include "MLModelRunner/PTModelRunner.h"

#include "MLModelRunner/MLModelRunner.h"
#include "SerDes/TensorSpec.h"
// #include "SerDes/baseSerDes.h"
#include "SerDes/pytorchSerDes.h"
#include "llvm/Support/ErrorHandling.h"
#include <torch/torch.h>
#include <torch/csrc/inductor/aoti_runner/model_container_runner_cpu.h> // or model_container_runner_cuda.h for CUDA

#include <memory>
#include <vector>

using TensorVec = std::vector<torch::Tensor>;

namespace MLBridge
{

    PTModelRunner::PTModelRunner(const std::string &modelPath, llvm::LLVMContext &Ctx)
        : MLModelRunner(MLModelRunner::Kind::PTAOT, BaseSerDes::Kind::Pytorch, &Ctx)
    {
        this->SerDes = new PytorchSerDes();

        c10::InferenceMode mode;
        this->CompiledModel = new torch::inductor::AOTIModelContainerRunnerCpu(modelPath);
    }



  void *PTModelRunner::evaluateUntyped()
  {

    if ((*reinterpret_cast<TensorVec*>(this->SerDes->getRequest())).empty())
    {
      llvm::errs() << "Input vector is empty.\n";
      return nullptr;
    }

    try
    {
      
      std::vector<torch::Tensor> *outputTensors = reinterpret_cast<std::vector<torch::Tensor>*>(this->SerDes->getResponse());
      auto outputs = reinterpret_cast<torch::inductor::AOTIModelContainerRunnerCpu*>(this->CompiledModel)->run((*reinterpret_cast<TensorVec*>(this->SerDes->getRequest())));
      for (auto i = outputs.begin(); i != outputs.end(); ++i)
        (*(outputTensors)).push_back(*i);
      void *rawData = this->SerDes->deserializeUntyped(outputTensors);
      return rawData;
    }
    catch (const c10::Error &e)
    {
      llvm::errs() << "Error during model evaluation: " << e.what() << "\n";
      return nullptr;
    }
  }

  template <typename U, typename T, typename... Types>
  void PTModelRunner::populateFeatures(const std::pair<U, T> &var1,
                                       const std::pair<U, Types> &...var2)
  {
    SerDes->setFeature(var1.first, var1.second);
    PTModelRunner::populateFeatures(var2...);
  }

} // namespace MLBridge
