# general structure:

# - should be a standalone automatic converter program
# - also includes tests

cmake_minimum_required(VERSION 3.11)
project(onnx-mlpack C CXX)

include(CMake/mlpack.cmake)
set(CMAKE_MODULE_PATH ${CMAKE_MODULKE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/CMake")

option(BUILD_TESTS "Build tests." ON)
option(DOWNLOAD_DEPENDENCIES "Download missing dependencies (including mlpack)."
    ON)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif ()

if (DOWNLOAD_DEPENDENCIES)
  fetch_mlpack(ON)
else ()
  find_mlpack()
endif ()
include_directories(${MLPACK_INCLUDE_DIRS})

# So that the ONNX protobuf file gets included correctly.
# TODO: integrate into the converter itself before include
add_definitions(-DONNX_ML=1)

find_package(ONNX REQUIRED)
include_directories(${ONNX_INCLUDE_DIRS})
message(STATUS "ONNX libraries: ${ONNX_LIBRARIES}")

# TODO: install headers to relevant location

#add_executable(mlpack_onnx_converter "src/mlpack_onnx_converter.cpp")
include_directories(include/)
add_subdirectory(tests/)
