cmake_minimum_required(VERSION 3.16)
project(TPUBenchmarkCpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(PkgConfig REQUIRED)

# Find TensorFlow Lite
find_path(TENSORFLOW_LITE_INCLUDE_DIR 
    NAMES tensorflow/lite/interpreter.h
    PATHS 
        /usr/local/include
        /usr/include
        ${CMAKE_SOURCE_DIR}/tensorflow
)

find_library(TENSORFLOW_LITE_LIB
    NAMES tensorflowlite tensorflow-lite
    PATHS
        /usr/local/lib
        /usr/lib
        /usr/lib/aarch64-linux-gnu
)

# Find Edge TPU library
find_path(EDGETPU_INCLUDE_DIR
    NAMES edgetpu.h
    PATHS
        /usr/local/include
        /usr/include
        ${CMAKE_SOURCE_DIR}/edgetpu/include
)

find_library(EDGETPU_LIB
    NAMES edgetpu
    PATHS
        /usr/local/lib
        /usr/lib
        /usr/lib/aarch64-linux-gnu
)

# Check if libraries were found
if(NOT TENSORFLOW_LITE_LIB)
    message(WARNING "TensorFlow Lite library not found. Trying alternative approach...")
    # Alternative: build minimal version without TF Lite
    add_definitions(-DNO_TENSORFLOW_LITE)
endif()

if(NOT EDGETPU_LIB)
    message(WARNING "Edge TPU library not found. Building without TPU support...")
    add_definitions(-DNO_EDGETPU)
endif()

# Include directories
include_directories(${TENSORFLOW_LITE_INCLUDE_DIR})
include_directories(${EDGETPU_INCLUDE_DIR})

# Create the executable
add_executable(plot_cpp plot_cpp.cpp)

# Link libraries
target_link_libraries(plot_cpp 
    ${TENSORFLOW_LITE_LIB}
    ${EDGETPU_LIB}
    pthread
)

# Compiler flags
target_compile_options(plot_cpp PRIVATE 
    -Wall -Wextra -O2
)

# Install target
install(TARGETS plot_cpp DESTINATION bin)
