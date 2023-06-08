cmake_minimum_required(VERSION 3.0)

# Set the C++ standard (optional)
set(CMAKE_CXX_STANDARD 17)

# Proto Source Files
# Update here if more proto files needed
set(PROTO_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/tokenize.proto
)

set(Protobuf_IMPORT_DIRS ${Protobuf_IMPORT_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})

# Find Protobuf installation
set(protobuf_MODULE_COMPATIBLE TRUE)
find_package(Protobuf CONFIG REQUIRED)
message(STATUS "Using Protobuf Version ${Protobuf_VERSION}")

set(_PROTOBUF_LIBPROTOBUF protobuf::libprotobuf)
set(_REFLECTION gRPC::grpc++_reflection)
set(_PROTOBUF_PROTOC $<TARGET_FILE:protobuf::protoc>)

# Find gRPC Install
find_package(gRPC CONFIG REQUIRED)
message(STATUS "Using gRPC Version ${gRPC_VERSION}")

set(_GRPC_GRPCPP gRPC::grpc++)
set(_GRPC_CPP_PLUGIN_EXECUTABLE $<TARGET_FILE:gRPC::grpc_cpp_plugin>)

# Update here if more proto files needed
set(targets tokenize)
set(proto_srcs)

foreach(_target IN LISTS targets)
set(hw_proto_srcs "${CMAKE_CURRENT_BINARY_DIR}/${_target}.pb.cc")
set(hw_proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/${_target}.pb.h")
set(hw_grpc_srcs "${CMAKE_CURRENT_BINARY_DIR}/${_target}.grpc.pb.cc")
set(hw_grpc_hdrs "${CMAKE_CURRENT_BINARY_DIR}/${_target}.grpc.pb.h")

set(proto_file "${CMAKE_CURRENT_SOURCE_DIR}/${_target}.proto")

get_filename_component(hw_proto ${proto_file} ABSOLUTE)
get_filename_component(hw_proto_path "${hw_proto}" PATH)

add_custom_command(
      OUTPUT "${hw_proto_srcs}" "${hw_proto_hdrs}" "${hw_grpc_srcs}" "${hw_grpc_hdrs}"
      COMMAND ${_PROTOBUF_PROTOC}
      ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
        --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
        -I "${hw_proto_path}"
        --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
        "${hw_proto}"
      DEPENDS "${hw_proto}")


LIST(APPEND proto_srcs ${hw_proto_srcs} ${hw_grpc_srcs})

endforeach()

add_library(tokenizeProto ${proto_srcs})
target_link_libraries(tokenizeProto
  PUBLIC
  protobuf::libprotobuf
  gRPC::grpc
  gRPC::grpc++
)

target_include_directories(tokenizeProto PUBLIC
  ./
  ${Protobuf_INCLUDE_DIRS}
  ${CMAKE_CURRENT_BINARY_DIR}
)

