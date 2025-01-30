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
JSON.cpp
)
target_compile_options (SerDesLib PUBLIC -fexceptions)

else()
  add_library(SerDesLib OBJECT TensorSpec.cpp jsonSerDes.cpp bitstreamSerDes.cpp protobufSerDes.cpp tensorflowSerDes.cpp JSON.cpp pytorchSerDes.cpp)

  add_library(SerDesCLib OBJECT TensorSpec.cpp jsonSerDes.cpp bitstreamSerDes.cpp JSON.cpp)
endif()
target_include_directories(SerDesLib PUBLIC ${TENSORFLOW_AOT_PATH}/include)
target_link_libraries(SerDesLib PRIVATE tf_xla_runtime)

find_package(Torch REQUIRED)
target_link_libraries(SerDesLib PUBLIC ${TORCH_LIBRARIES})
set_property(TARGET SerDesLib PROPERTY CXX_STANDARD 17)

target_compile_options(SerDesLib PRIVATE -frtti) 