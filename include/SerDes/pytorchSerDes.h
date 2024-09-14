#ifndef PYTORCH_SERIALIZER_H
#define PYTORCH_SERIALIZER_H

#include "SerDes/baseSerDes.h"
#include <iostream>
#include <vector>

#include <torch/script.h>
#include <torch/csrc/inductor/aoti_runner/model_container_runner_cpu.h>
#include <torch/torch.h>

namespace MLBridge {
class PytorchSerDes : public BaseSerDes {
public:
  PytorchSerDes() : BaseSerDes(SerDesKind::Pytorch) {}

  // Constructor that takes the model path and loads the model
  PytorchSerDes(const std::string &modelPath) : BaseSerDes(SerDesKind::Pytorch) {
    CompiledModel = new torch::inductor::AOTIModelContainerRunnerCpu(modelPath);
  }

  ~PytorchSerDes() { delete CompiledModel; }

  static bool classof(const BaseSerDes *S) {
    return S->getKind() == SerDesKind::Pytorch;
  }

#define SET_FEATURE(TYPE, _)                                                      \
  void setFeature(const std::string &, const TYPE) override;                 \
  void setFeature(const std::string &, const std::vector<TYPE> &) override;
  SUPPORTED_TYPES(SET_FEATURE)
#undef SET_FEATURE

  void setRequest(void *request) override {
    CompiledModel =
        reinterpret_cast<torch::inductor::AOTIModelContainerRunnerCpu *>(request);
  }

  void *getSerializedData() override;
  void cleanDataStructures() override;
  void *deserializeUntyped(void *data) override;

  // void addTensorToInput(const torch::Tensor &newTensor); // Add a tensor to the
                                                         // input tensor vector

private:
  std::vector<torch::Tensor> inputTensors; // Declaration of the input tensor
  std::vector<torch::Tensor> outputTensors; // Storage for the PyTorch output tensor
  torch::inductor::AOTIModelContainerRunnerCpu *CompiledModel;
};
} // namespace MLBridge

#endif