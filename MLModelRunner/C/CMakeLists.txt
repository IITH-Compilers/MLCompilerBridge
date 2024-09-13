find_package(Torch REQUIRED)
add_library(ModelRunnerCWrapper OBJECT PipeModelRunnerCWrapper.cpp ONNXModelRunnerCWrapper.cpp)
target_include_directories(ModelRunnerCWrapper PUBLIC ${TENSORFLOW_AOT_PATH}/include)
target_link_libraries(ModelRunnerCWrapper PUBLIC  "${TORCH_LIBRARIES}")
