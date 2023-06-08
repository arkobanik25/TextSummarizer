cmake_minimum_required(VERSION 3.0)
project(TextSummarizer C CXX)

# Set the C++ standard (optional)
set(CMAKE_CXX_STANDARD 17)

# Add the include directory to the include paths
include_directories(include third_party)

# Add the -g flag for debugging with GDB
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")

# Set the path to the shared library
set(LIBRARY_PATH ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

# Add all the source files in the src/ directory
file(GLOB SOURCES "src/*.cpp")

add_subdirectory(protos)

# Specify the executable and its source files
add_executable(TextSummarizer ${SOURCES})

# Link against the existing shared library
target_link_libraries(TextSummarizer PRIVATE 
    ${LIBRARY_PATH}/libonnxruntime.so
    tokenizeProto
    )
