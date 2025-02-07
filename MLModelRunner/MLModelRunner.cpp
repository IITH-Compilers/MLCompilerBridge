#include "MLModelRunner/MLModelRunner.h"

// #include "SerDes/baseSerDes.h"
#include "SerDes/bitstreamSerDes.h"
#include "SerDes/jsonSerDes.h"


#ifndef C_LIBRARY
#include "SerDes/protobufSerDes.h"
#include "SerDes/tensorflowSerDes.h"
#include "SerDes/pytorchSerDes.h" 
#endif

using namespace MLBridge;

/// MLModelRunner - The main interface for interacting with the ML models.
 void MLModelRunner::initSerDes() {
    switch (SerDesType) {
    case BaseSerDes::Kind::Json:
      SerDes = std::make_unique<JsonSerDes>();
      break;
    case BaseSerDes::Kind::Bitstream:
      SerDes = std::make_unique<BitstreamSerDes>();
      break;
    case BaseSerDes::Kind::Pytorch:
      SerDes = std::make_unique<PytorchSerDes>();
      break;
#ifndef C_LIBRARY
    case BaseSerDes::Kind::Protobuf:
      SerDes = std::make_unique<ProtobufSerDes>();
      break;
    case BaseSerDes::Kind::Tensorflow:
      SerDes = std::make_unique<TensorflowSerDes>();
      break;
#endif
    case BaseSerDes::Kind::Unknown:
      SerDes = nullptr;
      break;
    }

 }