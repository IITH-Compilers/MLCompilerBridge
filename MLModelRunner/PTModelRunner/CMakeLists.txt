if(LLVM_MLBRIDGE)
  add_llvm_library(PTModelRunnerLib PTModelRunner.cpp)
else()
  add_library(PTModelRunnerLib OBJECT PTModelRunner.cpp)
endif(LLVM_MLBRIDGE)

find_package(Torch REQUIRED)
target_link_libraries(PTModelRunnerLib PRIVATE ${TORCH_LIBRARIES})
target_compile_options (PTModelRunnerLib PRIVATE -fexceptions)

# message("HERE FIND TORCH ${TORCH_INCLUDE_DIRS}")
# find_package(Torch REQUIRED)
target_include_directories(PTModelRunnerLib PRIVATE ${TORCH_INCLUDE_DIRS})