find_package(Torch REQUIRED)

add_subdirectory(pytorchSerDes)

set(protobuf_MODULE_COMPATIBLE TRUE)
find_package(Protobuf CONFIG REQUIRED)

if(LLVM_MLBRIDGE)
add_llvm_library(SerDesLib
TensorSpec.cpp
jsonSerDes.cpp
bitstreamSerDes.cpp
protobufSerDes.cpp
tensorflowSerDes.cpp
JSON.cpp
)

else()
  add_library(SerDesLib OBJECT TensorSpec.cpp jsonSerDes.cpp bitstreamSerDes.cpp protobufSerDes.cpp tensorflowSerDes.cpp JSON.cpp )
endif()

target_link_libraries(SerDesLib PUBLIC PyTorchSerDesLib)

target_include_directories(SerDesLib PUBLIC ${TENSORFLOW_AOT_PATH}/include)
target_link_libraries(SerDesLib PRIVATE tf_xla_runtime)
