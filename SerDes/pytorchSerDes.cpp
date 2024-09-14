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
    inputTensors.push_back(tensor.clone());
}

void PytorchSerDes::setFeature(const std::string &Name, const long Value) {
    auto tensor = torch::tensor({Value}, torch::kInt64);
    inputTensors.push_back(tensor.clone());
}

void PytorchSerDes::setFeature(const std::string &Name, const float Value) {
    auto tensor = torch::tensor({Value}, torch::kFloat32);
    inputTensors.push_back(tensor.clone());
}

void PytorchSerDes::setFeature(const std::string &Name, const double Value) {
    auto tensor = torch::tensor({Value}, torch::kFloat64);
    inputTensors.push_back(tensor.clone());
}

void PytorchSerDes::setFeature(const std::string &Name, const std::string Value) {
    std::vector<int8_t> encoded_str(Value.begin(), Value.end());
    auto tensor = torch::tensor(encoded_str, torch::kInt8);
    inputTensors.push_back(tensor.clone());
}

void PytorchSerDes::setFeature(const std::string &Name, const bool Value) {
    auto tensor = torch::tensor({Value}, torch::kBool);
    inputTensors.push_back(tensor.clone());
}

void PytorchSerDes::setFeature(const std::string &Name, const std::vector<int> &Value) {
    auto tensor = torch::tensor(Value, torch::kInt32);
    inputTensors.push_back(tensor.clone());
}

void PytorchSerDes::setFeature(const std::string &Name, const std::vector<long> &Value) {
    auto tensor = torch::tensor(Value, torch::kInt64);
    inputTensors.push_back(tensor.clone());
}

void PytorchSerDes::setFeature(const std::string &Name, const std::vector<float> &Value) {
    auto tensor = torch::tensor(Value, torch::kFloat32);
    inputTensors.push_back(tensor.clone());
}

void PytorchSerDes::setFeature(const std::string &Name, const std::vector<double> &Value) {
    auto tensor = torch::tensor(Value, torch::kFloat64);
    inputTensors.push_back(tensor.clone());
}

void PytorchSerDes::setFeature(const std::string &Name, const std::vector<std::string> &Value) {
    std::vector<int8_t> flat_vec;
    for (const auto& str : Value) {
        flat_vec.insert(flat_vec.end(), str.begin(), str.end());
        flat_vec.push_back('\0'); // Null-terminate each string
    }
    auto tensor = torch::tensor(flat_vec, torch::kInt8);
    inputTensors.push_back(tensor.clone());
}

void PytorchSerDes::setFeature(const std::string &Name, const std::vector<bool> &Value) {
    std::vector<uint8_t> bool_vec(Value.begin(), Value.end());
    auto tensor = torch::tensor(bool_vec, torch::kUInt8);
    inputTensors.push_back(tensor.clone());
}

// void PytorchSerDes::setRequest(void *Request) {
//     CompiledModel = reinterpret_cast<torch::inductor::AOTIModelContainerRunnerCpu *>(Request);
// }

void PytorchSerDes::cleanDataStructures() {
    inputTensors.clear(); // Clear the input vector
}

void *PytorchSerDes::deserializeUntyped(void *Data) {
    if (Data == nullptr) {
        return nullptr;
    }

    // Assume Data is a pointer to a vector of tensors
    std::vector<torch::Tensor> *serializedTensors = reinterpret_cast<std::vector<torch::Tensor> *>(Data);

    // Calculate total size for the deserialized data
    size_t totalSize = 0;
    for (const auto &tensor : *serializedTensors) {
        totalSize += tensor.numel() * tensor.element_size();
    }

    // Giving value to the MessageLength variable
    MessageLength = totalSize;

    // Allocate a contiguous block of memory for the output
    auto *output = new uint8_t[totalSize];
    uint8_t *outputPtr = output;

    // Copy tensor data into the contiguous block of memory
    for (const auto &tensor : *serializedTensors) {
        size_t tensorSize = tensor.numel() * tensor.element_size();

        if (tensor.dtype() == torch::kInt32) {
            // Handle int32 tensors
            std::memcpy(outputPtr, tensor.data_ptr<int32_t>(), tensorSize);
        } else if (tensor.dtype() == torch::kInt64) {
            // Handle int64 tensors
            std::memcpy(outputPtr, tensor.data_ptr<int64_t>(), tensorSize);
        } else if (tensor.dtype() == torch::kFloat32) {
            // Handle float32 tensors
            std::memcpy(outputPtr, tensor.data_ptr<float>(), tensorSize);
        } else if (tensor.dtype() == torch::kFloat64) {
            // Handle float64 tensors
            std::memcpy(outputPtr, tensor.data_ptr<double>(), tensorSize);
        } else if (tensor.dtype() == torch::kBool) {
            // Handle bool tensors
            std::memcpy(outputPtr, tensor.data_ptr<bool>(), tensorSize);
        } else if (tensor.dtype() == torch::kInt8) {
            // Handle int8 tensors (e.g., for strings)
            std::memcpy(outputPtr, tensor.data_ptr<int8_t>(), tensorSize);
        } else {
            llvm::errs() << "Unsupported tensor dtype.\n";
            delete[] output;
            return nullptr;
        }

        outputPtr += tensorSize;
    }
    return static_cast<void *>(output);
}


//=== Implementation of the methods ===//
void *PytorchSerDes::getSerializedData() {
    std::vector<torch::Tensor> serializedData = outputTensors;

    // Allocate memory for the output and copy the serialized data
    auto *output = new std::vector<torch::Tensor>(serializedData);
    return static_cast<void *>(output);
}



} // namespace MLBridge
