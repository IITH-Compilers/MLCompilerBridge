add_subdirectory(Utils)
set(PROTOS_DIRECTORY "" CACHE PATH "Path to the directory containing the proto files")
if(NOT PROTOS_DIRECTORY STREQUAL "")
  add_subdirectory(gRPCModelRunner)
endif()
add_subdirectory(ONNXModelRunner)
add_subdirectory(C)

# # For up-to-date instructions for installing the TFLite dependency, refer to
# # the bot setup script: https://github.com/google/ml-compiler-opt/blob/main/buildbot/buildbot_init.sh
# set(LLVM_HAVE_TFLITE "" CACHE BOOL "Use tflite")
# if (LLVM_HAVE_TFLITE)
#   find_package(tensorflow-lite REQUIRED)
# endif()

# # For up-to-date instructions for installing the Tensorflow dependency, refer to
# # the bot setup script: https://github.com/google/ml-compiler-opt/blob/main/buildbot/buildbot_init.sh
# # Specifically, assuming python3 is installed:
# # python3 -m pip install --upgrade pip && python3 -m pip install --user tf_nightly==2.3.0.dev20200528
# # Then set TENSORFLOW_AOT_PATH to the package install - usually it's ~/.local/lib/python3.7/site-packages/tensorflow
# #
set(TENSORFLOW_AOT_PATH "" CACHE PATH "Path to TensorFlow pip install dir")

if (NOT TENSORFLOW_AOT_PATH STREQUAL "")
  include_directories(${TENSORFLOW_AOT_PATH}/include)
  add_subdirectory(${TENSORFLOW_AOT_PATH}/xla_aot_runtime_src
    ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}/tf_runtime)
    if(LLVM_MLBRIDGE)
      install(TARGETS tf_xla_runtime EXPORT LLVMExports
        ARCHIVE DESTINATION lib${LLVM_LIBDIR_SUFFIX} COMPONENT tf_xla_runtime)
      set_property(GLOBAL APPEND PROPERTY LLVM_EXPORTS tf_xla_runtime)
    else()
      install(TARGETS tf_xla_runtime ARCHIVE DESTINATION ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
    endif()
endif()

add_library(ModelRunnerCLib OBJECT PipeModelRunner.cpp)
target_link_libraries(ModelRunnerCLib PUBLIC ModelRunnerCUtils ONNXModelRunnerLib)
target_compile_definitions(ModelRunnerCLib PRIVATE C_LIBRARY)

if(LLVM_MLBRIDGE)
  add_llvm_library(ModelRunnerLib PipeModelRunner.cpp)
else()
  add_library(ModelRunnerLib OBJECT PipeModelRunner.cpp)
endif(LLVM_MLBRIDGE)

target_link_libraries(ModelRunnerLib PUBLIC ModelRunnerUtils ONNXModelRunnerLib)

if(NOT PROTOS_DIRECTORY STREQUAL "")
  target_link_libraries(ModelRunnerLib PUBLIC gRPCModelRunnerLib)
endif()
set_property(TARGET ModelRunnerLib PROPERTY POSITION_INDEPENDENT_CODE 1)

find_package(Torch REQUIRED)
target_link_libraries(ModelRunnerLib PRIVATE ${TORCH_LIBRARIES})
target_compile_options (ModelRunnerLib PRIVATE -frtti -fexceptions)
