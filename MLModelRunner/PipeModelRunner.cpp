//===- PipeModelRunner.cpp - noop ML model runner   ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A runner that communicates with an external agent via 2 file descriptors.
//===----------------------------------------------------------------------===//
#include "MLModelRunner/PipeModelRunner.h"
#include "MLModelRunner/MLModelRunner.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include <cstddef>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

using namespace llvm;

static cl::opt<bool> DebugReply(
    "interactive-model-runner-echo-reply", cl::init(false), cl::Hidden,
    cl::desc("The PipeModelRunner will echo back to stderr "
             "the data received from the host (for debugging purposes)."));

PipeModelRunner::PipeModelRunner(LLVMContext &Ctx, StringRef OutboundName,
                                 StringRef InboundName,
                                 BaseSerializer::Kind SerializerType)
    : MLModelRunner(Ctx, Kind::Pipe, SerializerType),
      InEC(sys::fs::openFileForRead(InboundName, Inbound)) {
  this->InboundName = InboundName.str();
  errs() << "InboundName: " << InboundName.str() << "\n";
  if (InEC) {
    Ctx.emitError("Cannot open inbound file: " + InEC.message());
    return;
  }
  {
    OutStream = std::make_unique<raw_fd_ostream>(OutboundName, OutEC);
    if (OutEC) {
      Ctx.emitError("Cannot open outbound file: " + OutEC.message());
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

void PipeModelRunner::send(void *data) {
  // TODO: send data size first (a hack to send protbuf data properly)
  auto dataString = reinterpret_cast<std::string *>(data);
  size_t message_length = dataString->size();
  const char *message_length_ptr =
      reinterpret_cast<const char *>(&message_length);
  OutStream->write(message_length_ptr, sizeof(size_t));
  OutStream->write(dataString->data(), dataString->size());
  OutStream->flush();
}

void *PipeModelRunner::receive() {

  size_t MessageLength = 0;
  size_t InsPoint = 0;
  char *Buff = (char *)(&MessageLength);
  const size_t LimitLength = sizeof(size_t);
  while (InsPoint < LimitLength) {
    auto ReadOrErr =
        ::sys::fs::readNativeFile(sys::fs::convertFDToNativeFile(Inbound),
                                  {Buff + InsPoint, LimitLength - InsPoint});
    if (ReadOrErr.takeError()) {
      Ctx.emitError("Failed reading from inbound file");
      break;
    }
    InsPoint += *ReadOrErr;
  }

  auto *Message = new std::string(MessageLength, '\0');
  InsPoint = 0;
  Buff = Message->data();
  const size_t Limit = Message->length();
  while (InsPoint < Limit) {
    auto ReadOrErr =
        ::sys::fs::readNativeFile(sys::fs::convertFDToNativeFile(Inbound),
                                  {Buff + InsPoint, Limit - InsPoint});
    if (ReadOrErr.takeError()) {
      Ctx.emitError("Failed reading from inbound file");
      break;
    }
    InsPoint += *ReadOrErr;
  }

  return Message;
}

void *PipeModelRunner::evaluateUntyped() {
  llvm::errs() << "In PipeModelRunner evaluateUntyped...\n";
  auto *data = Serializer->getSerializedData();
  send(data);
  auto *reply = receive();
  return Serializer->deserializeUntyped(reply);
}
