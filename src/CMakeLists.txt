# Add main onnx_mlpack_converter program.
add_executable(onnx_mlpack_converter "onnx_mlpack_converter.cpp")
target_precompile_headers(onnx_mlpack_converter PUBLIC
    <mlpack.hpp> <ensmallen.hpp>)
target_link_libraries(onnx_mlpack_converter PRIVATE ${MLPACK_LIBRARIES}
    ${ONNX_LIBRARIES})
# Make sure this gets built directly into the output directory.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/)

install(TARGETS onnx_mlpack_converter)
