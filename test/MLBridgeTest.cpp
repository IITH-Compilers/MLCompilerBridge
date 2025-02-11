//===----------------------------------------------------------------------===//
//
// Part of the MLCompilerBridge Project, under the Apache License v2.0 with LLVM
// Exceptions. See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "HelloMLBridge_Env.h"
#include "MLModelRunner/MLModelRunner.h"
#include "MLModelRunner/ONNXModelRunner/ONNXModelRunner.h"
#include "MLModelRunner/PipeModelRunner.h"
#include "MLModelRunner/TFModelRunner.h"
#include "MLModelRunner/Utils/DataTypes.h"
#include "MLModelRunner/Utils/MLConfig.h"
#include "MLModelRunner/gRPCModelRunner.h"
#include "ProtosInclude.h"
#include "llvm/Support/CommandLine.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <random>
#include <string>

#define debug_out                                                              \
  if (!silent)                                                                 \
  std::cout
using namespace grpc;

#define gRPCModelRunnerInit(datatype)                                          \
  increment_port(1);                                                           \
  MLBridgeTestgRPC_##datatype::Reply response;                                 \
  MLBridgeTestgRPC_##datatype::Request request;                                \
  MLRunner = std::make_unique<                                                 \
      gRPCModelRunner<MLBridgeTestgRPC_##datatype::MLBridgeTestService,        \
                      MLBridgeTestgRPC_##datatype::MLBridgeTestService::Stub,  \
                      MLBridgeTestgRPC_##datatype::Request,                    \
                      MLBridgeTestgRPC_##datatype::Reply>>(                    \
      server_address, &request, &response, nullptr);                           \
  MLRunner->setRequest(&request);                                              \
  MLRunner->setResponse(&response)

static llvm::cl::opt<std::string>
    cl_server_address("test-server-address", llvm::cl::Hidden,
                      llvm::cl::desc("Server address, format <ip>:<port>"),
                      llvm::cl::init(""));

static llvm::cl::opt<std::string>
    cl_pipe_name("test-pipe-name", llvm::cl::Hidden, llvm::cl::init(""),
                 llvm::cl::desc("Name for pipe file"));

static llvm::cl::opt<std::string>
    cl_onnx_path("onnx-model-path", llvm::cl::Hidden, llvm::cl::init(""),
                 llvm::cl::desc("Path to onnx model"));

static llvm::cl::opt<std::string> cl_test_config(
    "test-config", llvm::cl::Hidden,
    llvm::cl::desc("Method for communication with python model"));

static llvm::cl::opt<bool>
    silent("silent", llvm::cl::Hidden, llvm::cl::init(false),
           llvm::cl::desc("Only print errors when if set to true"));

namespace {
std::unique_ptr<MLModelRunner> MLRunner;
std::string basename;
BaseSerDes::Kind SerDesType;

std::string test_config;
std::string pipe_name;
std::string server_address;
std::string onnx_path;

// send value of type T1. Test received value of type T2 against expected value
template <typename T1, typename T2>
void testPrimitive(std::string label, T1 value, T2 expected) {
  std::pair<std::string, T1> p("request_" + label, value);
  MLRunner->populateFeatures(p);
  T2 out = MLRunner->evaluate<T2>();
  debug_out << "  " << label << " reply: " << out << "\n";
  if (std::abs(out - expected) > 10e-6) {
    std::cerr << "Error: Expected " << label << " reply: " << expected
              << ", Received: " << out << "\n";
    exit(1);
  }
}

template <typename T1, typename T2>
void testVector(std::string label, std::vector<T1> value,
                std::vector<T2> expected) {
  std::pair<std::string, std::vector<T1>> p("request_" + label, value);
  MLRunner->populateFeatures(p);
  T2 *out;
  size_t size;
  MLRunner->evaluate<T2 *>(out, size);
  std::vector<T2> reply(out, out + size);
  debug_out << "  " << label << " reply: ";
  int i = 0;
  for (auto x : reply) {
    debug_out << x << " ";
    if (std::abs(x - expected[i]) > 10e-6) {
      std::cerr << "Error: Expected " << label << " reply: " << expected[i]
                << ", Received: " << x << "\n";
      exit(1);
    }
    i++;
  }
  debug_out << "\n";
}

int testPipeBytes() {
  if (pipe_name == "") {
    std::cerr
        << "Pipe name must be specified via --test-pipe-name=<filename>\n";
    exit(1);
  }
  basename = "./" + pipe_name;
  SerDesType = BaseSerDes::Kind::Bitstream;
  MLRunner = std::make_unique<PipeModelRunner>(
      basename + ".out", basename + ".in", SerDesType, nullptr);
  testPrimitive<int, int>("int", 11, 12);
  testPrimitive<long, long>("long", 1234567890, 1234567891);
  testPrimitive<float, float>("float", 3.14, 4.14);
  testPrimitive<double, double>("double", 0.123456789123456789,
                                1.123456789123456789);
  testPrimitive<char, char>("char", 'a', 'b');
  testPrimitive<bool, bool>("bool", true, false);
  testVector<int, int>("vec_int", {11, 22, 33}, {12, 23, 34});
  testVector<long, long>("vec_long", {123456780, 222, 333},
                         {123456780, 123456781, 123456782});
  testVector<float, float>("vec_float", {11.1, 22.2, 33.3},
                           {1.11, 2.22, -3.33, 0});
  testVector<double, double>("vec_double",
                             {-1.1111111111, -2.2222222222, -3.3333333333},
                             {1.12345678912345670, -1.12345678912345671});
  return 0;
}

int testPipeJSON() {
  if (pipe_name == "") {
    std::cerr
        << "Pipe name must be specified via --test-pipe-name=<filename>\n";
    exit(1);
  }
  basename = "./" + pipe_name;
  SerDesType = BaseSerDes::Kind::Json;
  MLRunner = std::make_unique<PipeModelRunner>(
      basename + ".out", basename + ".in", SerDesType, nullptr);
  testPrimitive<int, IntegerType>("int", 11, 12);
  testPrimitive<long, IntegerType>("long", 1234567890, 12345);
  testPrimitive<float, RealType>("float", 3.14, 4.14);
  testPrimitive<double, RealType>("double", 0.123456789123456789,
                                  1.123456789123456789);
  testPrimitive<char, char>("char", 'a', 'b');
  testPrimitive<bool, bool>("bool", true, false);
  testVector<int, IntegerType>("vec_int", {11, 22, 33}, {12, 23, 34});
  testVector<long, IntegerType>("vec_long", {123456780, 222, 333},
                                {6780, 6781, 6782});
  testVector<float, RealType>("vec_float", {11.1, 22.2, 33.3},
                              {1.11, 2.22, -3.33, 0});
  testVector<double, RealType>("vec_double",
                               {-1.1111111111, -2.2222222222, -3.3333333333},
                               {1.12345678912345670, -1.12345678912345671});
  return 0;
}

void increment_port(int delta) {
  int split = server_address.find(":");
  int port = stoi(server_address.substr(split + 1));
  server_address =
      server_address.substr(0, split) + ":" + to_string(port + delta);
}

int testGRPC() {
  if (server_address == "") {
    std::cerr << "Server Address must be specified via "
                 "--test-server-address=\"<ip>:<port>\"\n";
    exit(1);
  }
  {
    gRPCModelRunnerInit(int);
    testPrimitive<int, int>("int", 11, 12);
  }
  {
    gRPCModelRunnerInit(long);
    testPrimitive<long, long>("long", 1234567890, 1234567891);
  }
  {
    gRPCModelRunnerInit(float);
    testPrimitive<float, float>("float", 3.14, 4.14);
  }
  {
    gRPCModelRunnerInit(double);
    testPrimitive<double, double>("double", 0.123456789123456789,
                                  1.123456789123456789);
  }
  increment_port(1);
  {
    gRPCModelRunnerInit(bool);
    testPrimitive<bool, bool>("bool", true, false);
  }
  {
    gRPCModelRunnerInit(vec_int);
    testVector<int, int>("vec_int", {11, 22, 33}, {12, 23, 34});
  }
  {
    gRPCModelRunnerInit(vec_long);
    testVector<long, long>("vec_long", {123456780, 222, 333},
                           {123456780, 123456781, 123456782});
  }
  {
    gRPCModelRunnerInit(vec_float);
    testVector<float, float>("vec_float", {11.1, 22.2, 33.3},
                             {1.11, 2.22, -3.33, 0});
  }
  {
    gRPCModelRunnerInit(vec_double);
    testVector<double, double>("vec_double",
                               {-1.1111111111, -2.2222222222, -3.3333333333},
                               {1.12345678912345670, -1.12345678912345671});
  }
  return 0;
}

class ONNXTest : public MLBridgeTestEnv {
public:
  int run(int expectedAction) {
    onnx_path = cl_onnx_path.getValue();
    if (onnx_path == "") {
      std::cerr << "ONNX model path must be specified via "
                   "--onnx-model-path=<filepath>\n";
      exit(1);
    }
    FeatureVector.clear();
    int n = 100;
    for (int i = 0; i < n; i++) {
      float delta = (float)(i - expectedAction) / n;
      FeatureVector.push_back(delta * delta);
    }

    Agent *agent = new Agent(onnx_path);
    std::map<std::string, Agent *> agents;
    agents["agent"] = agent;
    MLRunner = std::make_unique<ONNXModelRunner>(this, agents, nullptr);
    MLRunner->evaluate<int>();
    if (lastAction != expectedAction) {
      std::cerr << "Error: Expected action: " << expectedAction
                << ", Computed action: " << lastAction << "\n";
      exit(1);
    }
    return 0;
  }
};

} // namespace

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv);
  test_config = cl_test_config.getValue();

  if (test_config == "pipe-bytes") {
    pipe_name = cl_pipe_name.getValue();
    testPipeBytes();
  } else if (test_config == "pipe-json") {
    pipe_name = cl_pipe_name.getValue();
    testPipeJSON();
  } else if (test_config == "grpc") {
    server_address = cl_server_address.getValue();
    testGRPC();
  } else if (test_config == "onnx") {
    ONNXTest t;
    t.run(20);
  } else
    std::cerr << "--test-config must be provided from [pipe-bytes, pipe-json, "
                 "grpc, onnx]\n";
  return 0;
}
