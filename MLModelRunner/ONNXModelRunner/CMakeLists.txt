# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -DORT_NO_EXCEPTIONS")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra")

#onnxruntime providers
option(onnxruntime_USE_CUDA "Build with CUDA support" OFF)
option(onnxruntime_USE_OPENVINO "Build with OpenVINO support" OFF)
option(onnxruntime_USE_NNAPI_BUILTIN "Build with builtin NNAPI lib for Android NNAPI support" OFF)
option(onnxruntime_USE_DNNL "Build with DNNL support" OFF)
option(onnxruntime_USE_NUPHAR "Build with Nuphar" OFF)
option(onnxruntime_USE_TENSORRT "Build with TensorRT support" OFF)

set(ONNXRUNTIME_ROOTDIR "" CACHE PATH "Directory that contains ONNXRuntime" )
if(NOT ONNXRUNTIME_ROOTDIR)
  message( FATAL_ERROR "Set path to Onnx runtime in -DONNXRUNTIME_ROOTDIR variable" )
endif()

if(onnxruntime_USE_CUDA)
  add_definitions(-DUSE_CUDA)
endif()
if(onnxruntime_USE_OPENVINO)
  add_definitions(-DUSE_OPENVINO)
endif()
if(onnxruntime_USE_NNAPI_BUILTIN)
  add_definitions(-DUSE_NNAPI)
endif()
if(onnxruntime_USE_DNNL)
  add_definitions(-DUSE_DNNL)
endif()
if(onnxruntime_USE_NUPHAR)
  add_definitions(-DUSE_NUPHAR)
endif()
if(onnxruntime_USE_TENSORRT)
  add_definitions(-DUSE_TENSORRT)
endif()
if(onnxruntime_USE_DML)
  message("Enabling DML")
  add_definitions(-DUSE_DML)
endif()

if(LLVM_MLBRIDGE)
add_llvm_component_library(ONNXModelRunnerLib
onnx.cpp
agent.cpp
ONNXModelRunner.cpp
)
else()
add_library(ONNXModelRunnerLib OBJECT onnx.cpp
      agent.cpp
      ONNXModelRunner.cpp
    )
endif(LLVM_MLBRIDGE)

# target_include_directories(ONNXModelRunnerLib PUBLIC "${ONNXRUNTIME_ROOTDIR}/include" "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime/core/session")
target_include_directories(ONNXModelRunnerLib PUBLIC "${ONNXRUNTIME_ROOTDIR}/include" "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime/core/session" ${TENSORFLOW_AOT_PATH}/include)
target_link_directories(ONNXModelRunnerLib PUBLIC ${ONNXRUNTIME_ROOTDIR}/lib)
target_link_libraries(ONNXModelRunnerLib PRIVATE onnxruntime)

# find_package(Torch REQUIRED)
# target_link_libraries(ONNXModelRunnerLib PRIVATE ${TORCH_LIBRARIES})
target_compile_options (ONNXModelRunnerLib PUBLIC -frtti -fexceptions)
