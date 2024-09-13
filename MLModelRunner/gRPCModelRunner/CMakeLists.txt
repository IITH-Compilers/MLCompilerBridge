set(protobuf_MODULE_COMPATIBLE TRUE)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
find_package(Protobuf CONFIG REQUIRED)
set(_PROTOBUF_LIBPROTOBUF protobuf::libprotobuf)
set(_REFLECTION gRPC::grpc++_reflection)
if(CMAKE_CROSSCOMPILING)
  find_program(_PROTOBUF_PROTOC protoc)
else()
  set(_PROTOBUF_PROTOC $<TARGET_FILE:protobuf::protoc>)
endif()

# Find gRPC installation
# Looks for gRPCConfig.cmake file installed by gRPC's cmake installation.

find_package(gRPC 1.34.0 EXACT CONFIG REQUIRED)
message(STATUS "Using gRPC ${gRPC_VERSION}")

set(_GRPC_GRPCPP gRPC::grpc++)
if(CMAKE_CROSSCOMPILING)
  find_program(_GRPC_CPP_PLUGIN_EXECUTABLE grpc_cpp_plugin)
  find_program(_GRPC_PYTHON_PLUGIN_EXECUTABLE grpc_python_plugin)
else()
  set(_GRPC_CPP_PLUGIN_EXECUTABLE $<TARGET_FILE:gRPC::grpc_cpp_plugin>)
  set(_GRPC_PYTHON_PLUGIN_EXECUTABLE $<TARGET_FILE:gRPC::grpc_python_plugin>)
endif()

file(GLOB proto_list ${PROTOS_DIRECTORY}/*.proto)
set(proto_dir ${PROTOS_DIRECTORY})
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/include/grpc)


foreach(proto ${proto_list})
  get_filename_component(proto_name ${proto} NAME_WLE)
  file(GLOB cc_file ${CMAKE_CURRENT_SOURCE_DIR}/Service/${proto_name}/*.cc)
  set(cc_files ${cc_files} ${cc_file})
  file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/include/grpc/${proto_name})
  set(header_files ${header_files} "${CMAKE_BINARY_DIR}/include/grpc/${proto_name}")
  set(proto_srcs_list ${proto_srcs_list} "${CMAKE_BINARY_DIR}/include/grpc/${proto_name}/${proto_name}.pb.cc")
  set(grpc_srcs_list ${grpc_srcs_list} "${CMAKE_BINARY_DIR}/include/grpc/${proto_name}/${proto_name}.grpc.pb.cc")
  add_custom_command(
      OUTPUT "${CMAKE_BINARY_DIR}/include/grpc/${proto_name}/${proto_name}.pb.cc"
             "${CMAKE_BINARY_DIR}/include/grpc/${proto_name}/${proto_name}.pb.h"
             "${CMAKE_BINARY_DIR}/include/grpc/${proto_name}/${proto_name}.grpc.pb.cc"
             "${CMAKE_BINARY_DIR}/include/grpc/${proto_name}/${proto_name}.grpc.pb.h"
      COMMAND ${_PROTOBUF_PROTOC}
      ARGS --grpc_out "${CMAKE_BINARY_DIR}/include/grpc/${proto_name}"
      --cpp_out "${CMAKE_BINARY_DIR}/include/grpc/${proto_name}"
      -I "${proto_dir}"
      --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
      "${proto}"
      DEPENDS "${proto}"
  )
endforeach()

set( PYTHON_UTILITIES_DIRECTORY "" CACHE PATH "Path to the directory containing the python utilities")

if(PYTHON_UTILITIES_DIRECTORY STREQUAL "")
  set(PYTHON_UTILITIES_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/Python-Utilities)
endif()

file(MAKE_DIRECTORY ${PYTHON_UTILITIES_DIRECTORY})
foreach(proto ${proto_list})
  get_filename_component(proto_name ${proto} NAME_WLE)
  set(proto_python_srcs_list ${proto_python_srcs_list} "${PYTHON_UTILITIES_DIRECTORY}/${proto_name}_pb2.py")

  add_custom_command(
      OUTPUT "${PYTHON_UTILITIES_DIRECTORY}/${proto_name}_pb2.py"
      COMMAND ${_PROTOBUF_PROTOC}
      ARGS --grpc_out "${PYTHON_UTILITIES_DIRECTORY}/"
      --python_out "${PYTHON_UTILITIES_DIRECTORY}/"
      -I "${proto_dir}"
      --plugin=protoc-gen-grpc="${_GRPC_PYTHON_PLUGIN_EXECUTABLE}"
      "${proto}"
      DEPENDS "${proto}"
  )
endforeach()

# Building the library
if(LLVM_MLBRIDGE)
  add_llvm_component_library(gRPCModelRunnerLib
      ${cc_files}
      ${proto_srcs_list}
      ${grpc_srcs_list}
      ${proto_python_srcs_list}
  )
else()
  add_library(gRPCModelRunnerLib
      ${cc_files}
      ${proto_srcs_list}
      ${grpc_srcs_list}
      ${proto_python_srcs_list}
  )
endif(LLVM_MLBRIDGE)

target_link_libraries(gRPCModelRunnerLib
	  PRIVATE ${_REFLECTION}
	          ${_GRPC_GRPCPP}
	          ${_PROTOBUF_LIBPROTOBUF}
)

target_include_directories(gRPCModelRunnerLib
  PRIVATE ${CMAKE_BINARY_DIR}/include/Service
  PUBLIC ${Protobuf_INCLUDE_DIRS}
)
