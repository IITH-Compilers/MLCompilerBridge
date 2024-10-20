//===----------------------------------------------------------------------===//
//
// Part of the MLCompilerBridge Project, under the Apache License v2.0 with LLVM
// Exceptions. See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "MLModelRunner/MLModelRunner.h"
#include "MLModelRunner/PTModelRunner.h"
// #include "MLModelRunner/TFModelRunner.h"
#include "SerDes/pytorchSerDes.h"
#include "llvm/Support/CommandLine.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <random>
#include <string>
#include <iostream>
#include <chrono>

using namespace MLBridge;
using namespace std;

int main(int argc, char **argv) {
 
  printf("Starting PTModelRunner test...\n");
    llvm::LLVMContext Ctx;

    string modelPath = "model.so";
    unique_ptr<PTModelRunner> runner = make_unique<PTModelRunner>(modelPath, Ctx);

    // Create some input data
    vector<float> inputData(10, 0.5f); // Example input with size 10

    basic_string<char> input_str = "input";
    pair< string, vector<float>& > inputPair = make_pair(input_str, ref(inputData));
    
    runner->populateFeatures(inputPair);

    void* result = runner->evaluateUntyped();

    if (result) {
        cout << "Model evaluation succeeded." << endl;
        vector<float> *output = reinterpret_cast<vector<float> *>(result);
        cout << "Output: ";
        for (auto &v : *output) {
            cout << v << " ";
        }
        cout << endl;

    } else {
        cerr << "Model evaluation failed." << endl;
    }
  return 0;
}
