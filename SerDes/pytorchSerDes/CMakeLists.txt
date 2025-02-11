find_package(Torch REQUIRED)

# set(protobuf_MODULE_COMPATIBLE TRUE)
# find_package(Protobuf CONFIG REQUIRED)

if(LLVM_MLBRIDGE)
add_llvm_library(PyTorchSerDesLib
pytorchSerDes.cpp
)

else()
  add_library(PyTorchSerDesLib OBJECT pytorchSerDes.cpp)

  # add_library(SerDesCLib OBJECT TensorSpec.cpp jsonSerDes.cpp bitstreamSerDes.cpp JSON.cpp)
endif()

target_link_libraries(PyTorchSerDesLib PRIVATE ${TORCH_LIBRARIES})
target_compile_options(PyTorchSerDesLib PRIVATE -fexceptions)
target_include_directories(PyTorchSerDesLib PUBLIC ${TORCH_INCLUDE_DIRS})
