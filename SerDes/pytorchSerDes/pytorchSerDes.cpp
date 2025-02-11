//=== MLCompilerBridge/SerDes/pytorchSerDes.cpp - Pytorch SerDes Implementation ---*- C++ -*-===//
//
// Part of the MLCompilerBridge Project
//
//===------------------===//

#include "SerDes/pytorchSerDes.h"
#include "SerDes/baseSerDes.h"
#include <torch/torch.h>
#include <torch/csrc/inductor/aoti_runner/model_container_runner_cpu.h>

using TensorVec = std::vector<torch::Tensor>;

namespace MLBridge {

PytorchSerDes::PytorchSerDes() : BaseSerDes(BaseSerDes::Kind::Pytorch)  {
    // inputTensors = std::make_shared<std::vector<torch::Tensor>>();
    // outputTensors = new std::vector<torch::Tensor>();
    
    // RequestVoid = std::make_shared<std::vector<torch::Tensor>>();
    RequestVoid = new TensorVec();
    ResponseVoid = new TensorVec();
}

void PytorchSerDes::setFeature(const std::string &Name, const int Value) {
    auto tensor = torch::tensor({Value}, torch::kInt32);
    reinterpret_cast<TensorVec*>(this->RequestVoid)->push_back(tensor.clone());
}

void PytorchSerDes::setFeature(const std::string &Name, const long Value) {
    auto tensor = torch::tensor({Value}, torch::kInt64);
    reinterpret_cast<TensorVec*>(this->RequestVoid)->push_back(tensor.clone());
}

void PytorchSerDes::setFeature(const std::string &Name, const float Value) {
    auto tensor = torch::tensor({Value}, torch::kFloat32);
    reinterpret_cast<TensorVec*>(this->RequestVoid)->push_back(tensor.clone());
}

void PytorchSerDes::setFeature(const std::string &Name, const double Value) {
    auto tensor = torch::tensor({Value}, torch::kFloat64);
    reinterpret_cast<TensorVec*>(this->RequestVoid)->push_back(tensor.clone());
}

void PytorchSerDes::setFeature(const std::string &Name, const std::string Value) {
    std::vector<int8_t> encoded_str(Value.begin(), Value.end());
    auto tensor = torch::tensor(encoded_str, torch::kInt8);
    reinterpret_cast<TensorVec*>(this->RequestVoid)->push_back(tensor.clone());
}

void PytorchSerDes::setFeature(const std::string &Name, const bool Value) {
    auto tensor = torch::tensor({Value}, torch::kBool);
    reinterpret_cast<TensorVec*>(this->RequestVoid)->push_back(tensor.clone());
}

void PytorchSerDes::setFeature(const std::string &Name, const std::vector<int> &Value) {
    auto tensor = torch::tensor(Value, torch::kInt32);
    reinterpret_cast<TensorVec*>(this->RequestVoid)->push_back(tensor.clone());
}

void PytorchSerDes::setFeature(const std::string &Name, const std::vector<long> &Value) {
    auto tensor = torch::tensor(Value, torch::kInt64);
    reinterpret_cast<TensorVec*>(this->RequestVoid)->push_back(tensor.clone());
}

void PytorchSerDes::setFeature(const std::string &Name, const std::vector<float> &Value) {
    auto tensor = torch::tensor(Value, torch::kFloat32);
    tensor = tensor.reshape({1, Value.size()});
    reinterpret_cast<TensorVec*>(this->RequestVoid)->push_back(tensor.clone());
}

void PytorchSerDes::setFeature(const std::string &Name, const std::vector<double> &Value) {
    auto tensor = torch::tensor(Value, torch::kFloat64);
    reinterpret_cast<TensorVec*>(this->RequestVoid)->push_back(tensor.clone());
}

void PytorchSerDes::setFeature(const std::string &Name, const std::vector<std::string> &Value) {
    std::vector<int8_t> flat_vec;
    for (const auto& str : Value) {
        flat_vec.insert(flat_vec.end(), str.begin(), str.end());
        flat_vec.push_back('\0'); // Null-terminate each string
    }
    auto tensor = torch::tensor(flat_vec, torch::kInt8);
    reinterpret_cast<TensorVec*>(this->RequestVoid)->push_back(tensor.clone());
}

void PytorchSerDes::setFeature(const std::string &Name, const std::vector<bool> &Value) {
    std::vector<uint8_t> bool_vec(Value.begin(), Value.end());
    auto tensor = torch::tensor(bool_vec, torch::kUInt8);
    reinterpret_cast<TensorVec*>(this->RequestVoid)->push_back(tensor.clone());
}

// void PytorchSerDes::setRequest(void *Request) {
//     CompiledModel = reinterpret_cast<torch::inductor::AOTIModelContainerRunnerCpu *>(Request);
// }

void PytorchSerDes::cleanDataStructures() {
    reinterpret_cast<TensorVec*>(this->RequestVoid)->clear(); // Clear the input vector
}

void *PytorchSerDes::deserializeUntyped(void *Data) {
    if (Data == nullptr) {
        return nullptr;
    }

    // Assume Data is a pointer to a vector of tensors
    std::vector<torch::Tensor> *serializedTensors = reinterpret_cast<TensorVec *>(Data);

    if (serializedTensors->empty()) {
        return nullptr;
    }

    auto type_vect = serializedTensors->at(0).dtype();

    if (type_vect == torch::kInt32) {
        return copyTensorToVect<int32_t>(Data);
    } 
    else if (type_vect == torch::kInt64) {
        return copyTensorToVect<int64_t>(Data);
    } 
    else if (type_vect == torch::kFloat32) {
        return copyTensorToVect<float>(Data);
    } 
    else if (type_vect == torch::kFloat64) {
        return copyTensorToVect<double>(Data);
    } 
    else if (type_vect == torch::kBool) {
        return copyTensorToVect<bool>(Data);
    } 
    else if (type_vect == torch::kInt8) {
        return copyTensorToVect<int8_t>(Data);
    } 
    else {
        llvm::errs() << "Unsupported tensor dtype.\n";
        return nullptr;
    }
}

void *PytorchSerDes::getSerializedData() {
    return this->ResponseVoid;   // TODO - check
    // TensorVec serializedData = *reinterpret_cast<TensorVec*>(this->ReponseVoid);

    // // Allocate memory for the output and copy the serialized data
    // auto *output = new TensorVec(serializedData);
    // return static_cast<void *>(output);
}

 template <typename T>
    std::vector<T> *PytorchSerDes::copyTensorToVect(void *serializedTensors) {
        auto *ret = new std::vector<T>();
        for (const auto &tensor : *reinterpret_cast<TensorVec*>(serializedTensors)) {
            ret->insert(ret->end(), tensor.data_ptr<T>(), tensor.data_ptr<T>() + tensor.numel());
        }
        return ret;
    }

void *PytorchSerDes::getRequest() { return this->RequestVoid; }
void *PytorchSerDes::getResponse() { return this->ResponseVoid; }


} // namespace MLBridge