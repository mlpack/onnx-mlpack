# Try to find the ONNX headers and runtime libraries required for the mlpack
# ONNX converter.  Once done, the following variables will be defined:
#
# - ONNX_INCLUDE_DIRS: the directories that must be included; a list including
#     dependencies (protobuf)
# - ONNX_LIBRARIES: the libraries that must be linked against (including
#     protobuf)

if(ONNX_FOUND)
  return()
endif()

# Find protobuf (dependency).
find_package(Protobuf)
if (NOT Protobuf_FOUND)
  return()
endif ()

# Find the onnx_pb.h header.
find_path(ONNX_INCLUDE_DIR NAMES onnx_pb.h
    PATHS /usr/include /usr/local/include /opt/local/include /opt/include)
# It could be in an onnx/ directory.
if (NOT ONNX_INCLUDE_DIR)
  find_path(ONNX_INCLUDE_DIR NAMES onnx/onnx_pb.h
      PATHS /usr/include /usr/local/include /opt/local/include /opt/include)
endif ()

# Extract the version.  First try to look for ONNX_VERSION_MAJOR, although this
# does not exist in older versions of ONNX:
#
# https://github.com/onnx/onnx/pull/7918
#
file(STRINGS "${ONNX_INCLUDE_DIR}/onnx/common/version.h" _ONNX_VERSION_MAJOR
    REGEX "^#define ONNX_VERSION_MAJOR")
if (NOT "${_ONNX_VERSION_MAJOR}" STREQUAL "")
  file(STRINGS "${ONNX_INCLUDE_DIR}/onnx/common/version.h" _ONNX_VERSION_MINOR
      REGEX "^#define ONNX_VERSION_MINOR")
  file(STRINGS "${ONNX_INCLUDE_DIR}/onnx/common/version.h" _ONNX_VERSION_PATCH
      REGEX "^#define ONNX_VERSION_PATCH")

  message(STATUS "version major ${_ONNX_VERSION_MAJOR}")
  message(STATUS "version minor ${_ONNX_VERSION_MINOR}")
  message(STATUS "version patch ${_ONNX_VERSION_PATCH}")

  string(REGEX REPLACE "^#define ONNX_VERSION_MAJOR[ \t]+([0-9]+)[ \t]*$" "\\1"
      ONNX_VERSION_MAJOR "${_ONNX_VERSION_MAJOR}")
  string(REGEX REPLACE "^#define ONNX_VERSION_MINOR[ \t]+([0-9]+)[ \t]*$" "\\1"
      ONNX_VERSION_MINOR "${_ONNX_VERSION_MINOR}")
  string(REGEX REPLACE "^#define ONNX_VERSION_PATCH[ \t]+([0-9]+)[ \t]*$" "\\1"
      ONNX_VERSION_PATCH "${_ONNX_VERSION_PATCH}")

  set(ONNX_VERSION
      "${ONNX_VERSION_MAJOR}.${ONNX_VERSION_MINOR}.${ONNX_VERSION_PATCH}")
else ()
  # In this case, we have to do the extraction by hand since the version is too
  # old for the macros.
  file(STRINGS "${ONNX_INCLUDE_DIR}/onnx/common/version.h"
      _ONNX_VERSION_CONTENTS REGEX "LAST_RELEASE_VERSION")
  string(REGEX REPLACE "^.*\"([0-9.]+)\".*$"
      "\\1" ONNX_VERSION "${_ONNX_VERSION_CONTENTS}")
endif ()

find_library(ONNX_LIBRARY NAMES onnx
    PATHS /usr/lib /usr/lib64 /usr/local/lib /usr/local/lib64 /opt/local/lib/
    ENV LIBRARY_PATH ENV LD_LIBRARY_PATH)

find_library(ONNX_PROTO_LIBRARY NAMES onnx_proto
    PATHS /usr/lib /usr/lib64 /usr/local/lib /usr/local/lib64 /opt/local/lib/
    ENV LIBRARY_PATH ENV LD_LIBRARY_PATH)

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(ONNX
    REQUIRED_VARS ONNX_LIBRARY ONNX_PROTO_LIBRARY ONNX_INCLUDE_DIR
    VERSION_VAR ONNX_VERSION)

set(ONNX_INCLUDE_DIRS ${ONNX_INCLUDE_DIR} ${Protobuf_INCLUDE_DIRS})
set(ONNX_LIBRARIES ${ONNX_LIBRARY} ${ONNX_PROTO_LIBRARY} ${Protobuf_LIBRARIES})
mark_as_advanced(ONNX_INCLUDE_DIRS ONNX_LIBRARIES)
