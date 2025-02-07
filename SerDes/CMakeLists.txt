set(protobuf_MODULE_COMPATIBLE TRUE)
find_package(Protobuf CONFIG REQUIRED)

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
# target_compile_options (SerDesLib PUBLIC -fexceptions)

else()
  add_library(SerDesLib OBJECT TensorSpec.cpp jsonSerDes.cpp bitstreamSerDes.cpp protobufSerDes.cpp tensorflowSerDes.cpp JSON.cpp pytorchSerDes.cpp)
  # add_library(SerDesLib OBJECT TensorSpec.cpp jsonSerDes.cpp bitstreamSerDes.cpp JSON.cpp pytorchSerDes.cpp)

  # add_library(SerDesCLib OBJECT TensorSpec.cpp jsonSerDes.cpp bitstreamSerDes.cpp JSON.cpp)
endif()

target_include_directories(SerDesLib PUBLIC ${TENSORFLOW_AOT_PATH}/include)
target_link_libraries(SerDesLib PUBLIC tf_xla_runtime)

find_package(Torch REQUIRED)
target_link_libraries(SerDesLib PUBLIC ${TORCH_LIBRARIES})
target_compile_options(SerDesLib PUBLIC -frtti -fexceptions)

# Testing
message("PROTOBUF INCLUDES FROM ${Protobuf_INCLUDE_DIRS}")
target_include_directories(SerDesLib
 PRIVATE ${CMAKE_BINARY_DIR}/include/Service
 PUBLIC ${Protobuf_INCLUDE_DIRS}
)