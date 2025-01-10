//===----------------------------------------------------------------------===//
//
// Part of the MLCompilerBridge Project, under the Apache License v2.0 with LLVM
// Exceptions. See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MLModelRunner/MLModelRunner.h"
#include "MLModelRunner/PTModelRunner.h"
#include "SerDes/pytorchSerDes.h"
#include "llvm/Support/CommandLine.h"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <random>
#include <string>
#include <vector>

using namespace MLBridge;
using namespace std;

int main(int argc, char **argv)
{
  if (argc != 2)
  {
    cerr << "Usage: " << argv[0] << " <size>" << endl;
    return 1;
  }

  int sz = stoi(argv[1]);
  cout << "Running inference for size: " << sz << endl;

  // Create random input data of size sz
  vector<float> inputData(sz);
  random_device rd;
  mt19937 gen(rd());
  uniform_real_distribution<float> dis(0.0, 1.0);
  generate(inputData.begin(), inputData.end(), [&]()
           { return dis(gen); });

  llvm::LLVMContext Ctx;
  string modelPath = "models_pt/model_" + to_string(sz) + ".so";

  try
  {
    // Start measuring time just before loading the model
    auto start = chrono::high_resolution_clock::now();

    unique_ptr<PTModelRunner> runner = make_unique<PTModelRunner>(modelPath, Ctx);

    // Prepare the input
    basic_string<char> input_str = "input";
    pair<string, vector<float> &> inputPair = make_pair(input_str, ref(inputData));
    runner->populateFeatures(inputPair);

    // Perform inference
    void *result = runner->evaluateUntyped();

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> inferenceTime = end - start;

    // Check result
    if (result)
    {
      vector<float> *output = reinterpret_cast<vector<float> *>(result);
      cout << "Output: ";
      for (auto &v : *output)
      {
        cout << v << " ";
      }
      cout << endl;
    }
    else
    {
      cerr << "Model evaluation failed." << endl;
      return 1;
    }

    // // Log the results
    // ofstream resultsFile("results.csv", ios::app);
    // if (resultsFile.is_open())
    // {
    //   resultsFile << sz << "," << inferenceTime.count() << endl;
    //   resultsFile.close();
    // }
    // else
    // {
    //   cerr << "Failed to open results.csv" << endl;
    // }

    cout << "Inference completed in: " << inferenceTime.count() << " ms" << endl;
  }
  catch (const exception &ex)
  {
    cerr << "Error: " << ex.what() << endl;
    return 1;
  }

  return 0;
}
