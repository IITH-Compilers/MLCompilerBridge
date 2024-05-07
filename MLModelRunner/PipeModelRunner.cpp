//===- PipeModelRunner.cpp - Pipe based Model Runner ------------*- C++ -*-===//
//
// Part of the MLCompilerBridge Project, under the Apache License v2.0 with LLVM
// Exceptions. See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (Preliminary version adopted from InteractiveModelRunner.cpp of LLVM 17.X)
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the PipeModelRunner class to interface with an external ML
/// model during training and inference. The model is assumed to be running as
/// an external process and the communication is done via 2 file descriptors
/// using pipes.
///
//===----------------------------------------------------------------------===//

#include "MLModelRunner/PipeModelRunner.h"
#include "MLModelRunner/MLModelRunner.h"
#include "MLModelRunner/Utils/Debug.h"
#include <cstddef>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

#define DEBUG_TYPE "pipe-model-runner"

using namespace llvm;

namespace MLBridge {
PipeModelRunner::PipeModelRunner(StringRef OutboundName, StringRef InboundName,
                                 SerDesKind SerDesType, LLVMContext *Ctx)
    : MLModelRunner(Kind::Pipe, SerDesType, Ctx),
      InEC(sys::fs::openFileForRead(InboundName, Inbound)) {
  this->InboundName = InboundName.str();
  if (InEC) {
    int max_retries = 30, attempts = 0;
    double wait_seconds = 0.2, backoff_exp = 1.2;

    while (attempts < max_retries) {
      InEC = sys::fs::openFileForRead(InboundName, Inbound);
      if (InEC) {
        attempts++;
        std::cout << "Cannot open inbound file retrying! attempt: " << attempts
                  << std::endl;
        std::this_thread::sleep_for(
            std::chrono::duration<double>(wait_seconds));
        wait_seconds *= backoff_exp;
      } else {
        break;
      }
    }
    if (InEC) {
      auto message = "Cannot open inbound file: " + InEC.message();
      if (this->Ctx)
        this->Ctx->emitError(message);
      else
        std::cerr << message << std::endl;
      return;
    }
  }
  {
    OutStream = std::make_unique<raw_fd_ostream>(OutboundName, OutEC);
    if (OutEC) {
      auto message = "Cannot open outbound file: " + OutEC.message();
      if (this->Ctx)
        this->Ctx->emitError(message);
      else
        std::cerr << message << std::endl;
      return;
    }
  }
}

PipeModelRunner::~PipeModelRunner() {
  // close the file descriptors
  sys::fs::file_t FDAsOSHandle = sys::fs::convertFDToNativeFile(Inbound);
  sys::fs::closeFile(FDAsOSHandle);

  OutStream->close();
}

std::string PipeModelRunner::readNBytes(size_t N) {
  std::string OutputBuffer(N, '\0');
  char *Buff = OutputBuffer.data();
  size_t InsPoint = 0;
  const size_t Limit = N;
  while (InsPoint < Limit) {
    auto ReadOrErr = ::sys::fs::readNativeFile(
        sys::fs::convertFDToNativeFile(Inbound),
        {Buff + InsPoint, OutputBuffer.size() - InsPoint});
    if (ReadOrErr.takeError()) {
      if (this->Ctx)
        this->Ctx->emitError("Failed reading from inbound file");
      else
        std::cerr << "Failed reading from inbound file" << std::endl;
      break;
    }
    InsPoint += *ReadOrErr;
  }
  return OutputBuffer;
}

void PipeModelRunner::send(void *data) {
  // TODO: send data size first (a hack to send protbuf data properly)
  auto dataString = reinterpret_cast<std::string *>(data);
  size_t message_length = dataString->size();
  const char *message_length_ptr =
      reinterpret_cast<const char *>(&message_length);
  MLBRIDGE_DEBUG(std::cout << "Message length: " << message_length << "\n");
  MLBRIDGE_DEBUG(std::cout << "DataString.size(): " << dataString->size()
                           << "\n");
  OutStream->write(message_length_ptr, sizeof(size_t));
  OutStream->write(dataString->data(), dataString->size());
  OutStream->flush();
}

void *PipeModelRunner::receive() {
  MLBRIDGE_DEBUG(std::cout << "In PipeModelRunner receive...\n");
  auto hdr = readNBytes(8);
  MLBRIDGE_DEBUG(std::cout << "Read header...\n");
  size_t MessageLength = 0;
  memcpy(&MessageLength, hdr.data(), sizeof(MessageLength));
  // Read message
  auto OutputBuffer = new std::string(readNBytes(MessageLength));
  MLBRIDGE_DEBUG(std::cout << "OutputBuffer size: " << OutputBuffer->size()
                           << "\n";
                 std::cout << "OutputBuffer: " << *OutputBuffer << "\n");
  return OutputBuffer;
}

void *PipeModelRunner::evaluateUntyped() {
  MLBRIDGE_DEBUG(std::cout << "In PipeModelRunner evaluateUntyped...\n");
  auto *data = SerDes->getSerializedData();
  send(data);
  auto *reply = receive();
  MLBRIDGE_DEBUG(
      std::cout << "In PipeModelRunner::evaluateUntyped() received data...\n");
  return SerDes->deserializeUntyped(reply);
}

} // namespace MLBridge
