#ifndef PTMODELRUNNER_H
#define PTMODELRUNNER_H

#include "MLModelRunner/MLModelRunner.h"
#include "SerDes/TensorSpec.h"
// #include "SerDes/baseSerDes.h"
#include "SerDes/pytorchSerDes.h"
#include "llvm/Support/ErrorHandling.h"
// #include <torch/torch.h>
// #include <torch/csrc/inductor/aoti_runner/model_container_runner_cpu.h> // or model_container_runner_cuda.h for CUDA

#include <memory>
#include <vector>

namespace MLBridge
{

  /// PTModelRunner - PyTorch Compiled model implementation of the
  /// MLModelRunner. It uses an AOT-compiled model for efficient execution.
  class PTModelRunner final : public MLModelRunner
  {
  public:
    // New constructor that takes the model path as an input
    PTModelRunner(const std::string &modelPath, llvm::LLVMContext &Ctx);
    // {
    //   this->SerDes = new PytorchSerDes();

    //   c10::InferenceMode mode;
    //   this->CompiledModel = new torch::inductor::AOTIModelContainerRunnerCpu(modelPath);
    // }

    virtual ~PTModelRunner() = default;

    virtual void requestExit() override
    {
      llvm_unreachable("requestExit() is not supported in PTModelRunner");
    }

    static bool classof(const MLModelRunner *R)
    {
      return R->getKind() == MLModelRunner::Kind::PTAOT;
    }

    template <typename U, typename T, typename... Types>
    void populateFeatures(const std::pair<U, T> &var1,
                          const std::pair<U, Types> &...var2);

    void populateFeatures() {}

    void *evaluateUntyped() override;
    PytorchSerDes *SerDes;
    // Compiled model container added to the PTModelRunner
  private:
    void *CompiledModel; 
    // torch::inductor::AOTIModelContainerRunnerCpu *CompiledModel; 
  };

  // void *PTModelRunner::evaluateUntyped()
  // {

  //   if ((*(this->SerDes->inputTensors)).empty())
  //   {
  //     llvm::errs() << "Input vector is empty.\n";
  //     return nullptr;
  //   }

  //   try
  //   {
  //     auto outputs = this->CompiledModel->run((*(this->SerDes->inputTensors)));
  //     for (auto i = outputs.begin(); i != outputs.end(); ++i)
  //       (*(this->SerDes->outputTensors)).push_back(*i);
  //     void *rawData = this->SerDes->deserializeUntyped(this->SerDes->outputTensors);
  //     return rawData;
  //   }
  //   catch (const c10::Error &e)
  //   {
  //     llvm::errs() << "Error during model evaluation: " << e.what() << "\n";
  //     return nullptr;
  //   }
  // }

  // template <typename U, typename T, typename... Types>
  // void PTModelRunner::populateFeatures(const std::pair<U, T> &var1,
  //                                      const std::pair<U, Types> &...var2)
  // {
  //   SerDes->setFeature(var1.first, var1.second);
  //   PTModelRunner::populateFeatures(var2...);
  // }

} // namespace MLBridge

#endif // TFMODELRUNNER_H