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

# Shape inference does not succeed in all cases before ONNX 1.15.0.
find_package(ONNX 1.15.0 REQUIRED)
include_directories(${ONNX_INCLUDE_DIRS})

# If we are not using Visual Studio, use the GNU install directories module.
# Otherwise set the values manually.
if (NOT MSVC)
  include(GNUInstallDirs)
else ()
  set(CMAKE_INSTALL_BINDIR ${CMAKE_INSTALL_PREFIX}/bin)
  set(CMAKE_INSTALL_LIBDIR ${CMAKE_INSTALL_PREFIX}/lib)
  set(CMAKE_INSTALL_MANDIR ${CMAKE_INSTALL_PREFIX}/man)
  set(CMAKE_INSTALL_DOCDIR ${CMAKE_INSTALL_PREFIX}/share/doc/mlpack)
  set(CMAKE_INSTALL_INCLUDEDIR ${CMAKE_INSTALL_PREFIX}/include)
endif ()

include_directories(include/)
install(FILES include/onnx_mlpack.hpp DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
install(DIRECTORY include/onnx_mlpack/ DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/onnx_mlpack/")
add_subdirectory(src/)
add_subdirectory(tests/)
