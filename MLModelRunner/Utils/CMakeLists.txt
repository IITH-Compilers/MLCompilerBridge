if(LLVM_MLBRIDGE)
  add_llvm_component_library(ModelRunnerUtils
  MLConfig.cpp
  )
  target_include_directories(ModelRunnerUtils PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../include)

else()
  add_library(ModelRunnerCUtils OBJECT MLConfig.cpp)
  add_library(ModelRunnerUtils OBJECT MLConfig.cpp)
endif(LLVM_MLBRIDGE)
