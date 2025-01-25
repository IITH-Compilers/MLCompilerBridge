#ifndef PYTORCH_SERIALIZER_H
#define PYTORCH_SERIALIZER_H

#include "SerDes/baseSerDes.h"
#include <iostream>
#include <vector>

// #include <torch/script.h>
#include <torch/torch.h>
#include <torch/csrc/inductor/aoti_runner/model_container_runner_cpu.h>

namespace MLBridge {
class PytorchSerDes : public BaseSerDes {
public:
  // Constructor that takes the model path and loads the model
  PytorchSerDes() : BaseSerDes(SerDesKind::Pytorch) {
    inputTensors = std::make_shared<std::vector<torch::Tensor>>();
    outputTensors = new std::vector<torch::Tensor>();
  }

  ~PytorchSerDes() {}

  static bool classof(const BaseSerDes *S) {
    return S->getKind() == SerDesKind::Pytorch;
  }

#define SET_FEATURE(TYPE, _)                                                      \
  void setFeature(const std::string &, const TYPE) override;                 \
  void setFeature(const std::string &, const std::vector<TYPE> &) override;
  SUPPORTED_TYPES(SET_FEATURE)
#undef SET_FEATURE

  void setRequest(void *request){
  }

 template <typename T>
    std::vector<T> *copyTensorToVect(std::vector<torch::Tensor> *serializedTensors) {
        auto *ret = new std::vector<T>();
        for (const auto &tensor : *serializedTensors) {
            ret->insert(ret->end(), tensor.data_ptr<T>(), tensor.data_ptr<T>() + tensor.numel());
        }
        return ret;
    }
  

  void *getSerializedData() override;
  void cleanDataStructures() override;
  void *deserializeUntyped(void *data) override;

  std::vector<torch::Tensor> *outputTensors; // Storage for the PyTorch output tensor
  std::shared_ptr<std::vector<torch::Tensor>> inputTensors; // Declaration of the input tensor

};
} // namespace MLBridge

#endif