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
  if (ONNX_INCLUDE_DIR)
    set(ONNX_INCLUDE_DIR "${ONNX_INCLUDE_DIR}/onnx")
  endif ()
endif ()

find_library(ONNX_LIBRARY NAMES onnx
    PATHS /usr/lib /usr/lib64 /usr/local/lib /usr/local/lib64 /opt/local/lib/
    ENV LIBRARY_PATH ENV LD_LIBRARY_PATH)

find_library(ONNX_PROTO_LIBRARY NAMES onnx_proto
    PATHS /usr/lib /usr/lib64 /usr/local/lib /usr/local/lib64 /opt/local/lib/
    ENV LIBRARY_PATH ENV LD_LIBRARY_PATH)

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(ONNX DEFAULT_MSG
  ONNX_LIBRARY
  ONNX_PROTO_LIBRARY
  ONNX_INCLUDE_DIR)

set(ONNX_INCLUDE_DIRS ${ONNX_INCLUDE_DIR} ${Protobuf_INCLUDE_DIRS})
set(ONNX_LIBRARIES ${ONNX_LIBRARY} ${ONNX_PROTO_LIBRARY} ${Protobuf_LIBRARIES})
mark_as_advanced(ONNX_INCLUDE_DIRS ONNX_LIBRARIES)
