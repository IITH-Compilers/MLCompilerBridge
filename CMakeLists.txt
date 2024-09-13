cmake_minimum_required(VERSION 3.10)

project(MLCompilerBridge VERSION 0.0.2)
add_compile_options("$<$<CONFIG:${CMAKE_BUILD_TYPE}>:-UNDEBUG>")
set(protobuf_MODULE_COMPATIBLE TRUE)
find_package(Protobuf CONFIG REQUIRED)
set(protobuf_MODULE_COMPATIBLE TRUE)
find_package(Protobuf CONFIG REQUIRED)
find_package(Torch REQUIRED)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib CACHE PATH "Output directory for static libraries")

include_directories(${Protobuf_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/include)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

set (CMAKE_CXX_STANDARD 17)

option(LLVM_MLBRIDGE "MLCompilerBridge install for LLVM" OFF)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

string( TOLOWER "${CMAKE_BUILD_TYPE}" MLBRIDGE_CMAKE_BUILD_TYPE)
if(MLBRIDGE_CMAKE_BUILD_TYPE MATCHES debug)
  option(MLBRIDGE_DEBUG_MODE "Enable debug mode" ON)
else()
  option(MLBRIDGE_DEBUG_MODE "Enable debug mode" OFF)
endif()

if(NOT LLVM_MLBRIDGE)
  find_package(LLVM 10.0.0 REQUIRED CONFIG)
  include_directories(${LLVM_INCLUDE_DIRS})
  link_directories(${LLVM_LIBRARY_DIR})
endif()

if(MLBRIDGE_DEBUG_MODE)
  add_compile_definitions(DEBUG_MODE)
endif()

add_subdirectory(MLModelRunner)
add_subdirectory(SerDes)
if(MLBRIDGE_ENABLE_TEST)
  add_subdirectory(test)
endif()

if(LLVM_MLBRIDGE)
  include(AddLLVM)
  include(HandleLLVMOptions)
  include(LLVMDistributionSupport)

  add_llvm_library(LLVMMLBridge
  tools.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/include

  LINK_LIBS
  ModelRunnerLib
  SerDesLib
  )  

  target_link_libraries(LLVMMLBridge PRIVATE "${TORCH_LIBRARIES}")

  target_include_directories(LLVMMLBridge SYSTEM PUBLIC ${Protobuf_INCLUDE_DIRS} ${TENSORFLOW_AOT_PATH}/include)
  target_include_directories(LLVMMLBridge PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  install(TARGETS LLVMMLBridge DESTINATION lib)
  add_custom_command(TARGET LLVMMLBridge
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/CompilerInterface ${CMAKE_CURRENT_BINARY_DIR}/CompilerInterface
  )

else()
  llvm_map_components_to_libnames(llvm_libs support core irreader analysis TransformUtils)

  add_library(MLCompilerBridge STATIC tools.cpp)
  target_link_libraries(MLCompilerBridge PUBLIC SerDesLib ModelRunnerLib ONNXModelRunnerLib ${llvm_libs})
  set_target_properties(MLCompilerBridge PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
  set_property(TARGET MLCompilerBridge PROPERTY POSITION_INDEPENDENT_CODE 1)
  install(TARGETS MLCompilerBridge DESTINATION lib)


  add_library(MLCompilerBridgeC STATIC $<TARGET_OBJECTS:ModelRunnerCWrapper>)
  target_link_libraries(MLCompilerBridgeC PUBLIC SerDesCLib ModelRunnerCLib ONNXModelRunnerLib ${llvm_libs})
  target_link_libraries(MLCompilerBridge PUBLIC "${TORCH_LIBRARIES}")

  target_include_directories(MLCompilerBridgeC PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${LLVM_INCLUDE_DIRS})
  target_compile_features(MLCompilerBridgeC PRIVATE cxx_std_17)
  target_compile_definitions(MLCompilerBridgeC PRIVATE C_LIBRARY)
  set_property(TARGET MLCompilerBridgeC PROPERTY POSITION_INDEPENDENT_CODE 1)
  set_target_properties(MLCompilerBridgeC PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
  install(TARGETS MLCompilerBridgeC DESTINATION lib)
  if(MLBRIDGE_ENABLE_TEST)
    add_executable(MLCompilerBridgeTest $<TARGET_OBJECTS:MLBridgeCPPTest>)
    target_link_libraries(MLCompilerBridgeTest PUBLIC MLCompilerBridge)
    set_target_properties(MLCompilerBridgeTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    install(TARGETS MLCompilerBridgeTest DESTINATION bin)
  endif()
endif(LLVM_MLBRIDGE)

install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY CompilerInterface DESTINATION MLModelRunner/CompilerInterface)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/CompilerInterface DESTINATION ${CMAKE_BINARY_DIR}/MLModelRunner/)
