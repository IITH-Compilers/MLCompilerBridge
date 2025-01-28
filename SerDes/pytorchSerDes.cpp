//=== MLCompilerBridge/SerDes/pytorchSerDes.cpp - Pytorch SerDes Implementation ---*- C++ -*-===//
//
// Part of the MLCompilerBridge Project
//
//===------------------===//

#include "SerDes/pytorchSerDes.h"
#include "SerDes/baseSerDes.h"

namespace MLBridge {

void PytorchSerDes::setFeature(const std::string &Name, const int Value) {
    auto tensor = torch::tensor({Value}, torch::kInt32);
    this->inputTensors->push_back(tensor.clone());
}

void PytorchSerDes::setFeature(const std::string &Name, const long Value) {
    auto tensor = torch::tensor({Value}, torch::kInt64);
    this->inputTensors->push_back(tensor.clone());
}

void PytorchSerDes::setFeature(const std::string &Name, const float Value) {
    auto tensor = torch::tensor({Value}, torch::kFloat32);
    this->inputTensors->push_back(tensor.clone());
}

void PytorchSerDes::setFeature(const std::string &Name, const double Value) {
    auto tensor = torch::tensor({Value}, torch::kFloat64);
    this->inputTensors->push_back(tensor.clone());
}

void PytorchSerDes::setFeature(const std::string &Name, const std::string Value) {
    std::vector<int8_t> encoded_str(Value.begin(), Value.end());
    auto tensor = torch::tensor(encoded_str, torch::kInt8);
    this->inputTensors->push_back(tensor.clone());
}

void PytorchSerDes::setFeature(const std::string &Name, const bool Value) {
    auto tensor = torch::tensor({Value}, torch::kBool);
    this->inputTensors->push_back(tensor.clone());
}

void PytorchSerDes::setFeature(const std::string &Name, const std::vector<int> &Value) {
    auto tensor = torch::tensor(Value, torch::kInt32);
    this->inputTensors->push_back(tensor.clone());
}

void PytorchSerDes::setFeature(const std::string &Name, const std::vector<long> &Value) {
    auto tensor = torch::tensor(Value, torch::kInt64);
    this->inputTensors->push_back(tensor.clone());
}

void PytorchSerDes::setFeature(const std::string &Name, const std::vector<float> &Value) {
    auto tensor = torch::tensor(Value, torch::kFloat32);
    tensor = tensor.reshape({1, Value.size()});
    inputTensors->push_back(tensor.clone());
}

void PytorchSerDes::setFeature(const std::string &Name, const std::vector<double> &Value) {
    auto tensor = torch::tensor(Value, torch::kFloat64);
    this->inputTensors->push_back(tensor.clone());
}

void PytorchSerDes::setFeature(const std::string &Name, const std::vector<std::string> &Value) {
    std::vector<int8_t> flat_vec;
    for (const auto& str : Value) {
        flat_vec.insert(flat_vec.end(), str.begin(), str.end());
        flat_vec.push_back('\0'); // Null-terminate each string
    }
    auto tensor = torch::tensor(flat_vec, torch::kInt8);
    this->inputTensors->push_back(tensor.clone());
}

void PytorchSerDes::setFeature(const std::string &Name, const std::vector<bool> &Value) {
    std::vector<uint8_t> bool_vec(Value.begin(), Value.end());
    auto tensor = torch::tensor(bool_vec, torch::kUInt8);
    this->inputTensors->push_back(tensor.clone());
}

// void PytorchSerDes::setRequest(void *Request) {
//     CompiledModel = reinterpret_cast<torch::inductor::AOTIModelContainerRunnerCpu *>(Request);
// }

void PytorchSerDes::cleanDataStructures() {
    this->inputTensors->clear(); // Clear the input vector
}

void *PytorchSerDes::deserializeUntyped(void *Data) {
    if (Data == nullptr) {
        return nullptr;
    }

    // Assume Data is a pointer to a vector of tensors
    std::vector<torch::Tensor> *serializedTensors = reinterpret_cast<std::vector<torch::Tensor> *>(Data);

    if (serializedTensors->empty()) {
        return nullptr;
    }

    auto type_vect = serializedTensors->at(0).dtype();

    if (type_vect == torch::kInt32) {
        return copyTensorToVect<int32_t>(serializedTensors);
    } 
    else if (type_vect == torch::kInt64) {
        return copyTensorToVect<int64_t>(serializedTensors);
    } 
    else if (type_vect == torch::kFloat32) {
        return copyTensorToVect<float>(serializedTensors);
    } 
    else if (type_vect == torch::kFloat64) {
        return copyTensorToVect<double>(serializedTensors);
    } 
    else if (type_vect == torch::kBool) {
        return copyTensorToVect<bool>(serializedTensors);
    } 
    else if (type_vect == torch::kInt8) {
        return copyTensorToVect<int8_t>(serializedTensors);
    } 
    else {
        llvm::errs() << "Unsupported tensor dtype.\n";
        return nullptr;
    }
}

void *PytorchSerDes::getSerializedData() {
    std::vector<torch::Tensor> serializedData = *(this->outputTensors);

    // Allocate memory for the output and copy the serialized data
    auto *output = new std::vector<torch::Tensor>(serializedData);
    return static_cast<void *>(output);
}



} // namespace MLBridge
