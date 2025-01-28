find_package(Torch REQUIRED)
add_library(ModelRunnerCWrapper OBJECT PipeModelRunnerCWrapper.cpp ONNXModelRunnerCWrapper.cpp)
target_include_directories(ModelRunnerCWrapper PRIVATE ${TENSORFLOW_AOT_PATH}/include)
target_link_libraries(ModelRunnerCWrapper PRIVATE  "${TORCH_LIBRARIES}")
