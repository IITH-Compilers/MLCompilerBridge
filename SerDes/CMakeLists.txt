set(protobuf_MODULE_COMPATIBLE TRUE)
find_package(Protobuf CONFIG REQUIRED)
find_package(Torch REQUIRED)



if(LLVM_MLBRIDGE)
  add_llvm_library(SerDesLib
  TensorSpec.cpp
  jsonSerDes.cpp
  bitstreamSerDes.cpp
  protobufSerDes.cpp
  pytorchSerDes.cpp
  tensorflowSerDes.cpp
  )
  target_compile_definitions(SerDesLib PRIVATE LLVM_MLBRIDGE)
else()
  add_library(SerDesLib OBJECT TensorSpec.cpp jsonSerDes.cpp bitstreamSerDes.cpp protobufSerDes.cpp tensorflowSerDes.cpp pytorchSerDes.cpp)

  add_library(SerDesCLib OBJECT TensorSpec.cpp jsonSerDes.cpp bitstreamSerDes.cpp )
endif()
target_include_directories(SerDesLib PUBLIC ${TENSORFLOW_AOT_PATH}/include)
target_link_libraries(SerDesLib PRIVATE tf_xla_runtime)
target_link_libraries(SerDesLib PRIVATE "${TORCH_LIBRARIES}")
