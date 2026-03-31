# Add main onnx_mlpack_converter program.
add_executable(onnx_mlpack_converter "onnx_mlpack_converter.cpp")
target_precompile_headers(onnx_mlpack_converter PUBLIC
    <armadillo> <ensmallen.hpp>)
target_link_libraries(onnx_mlpack_converter PRIVATE ${MLPACK_LIBRARIES} ${ONNX_LIBRARIES})
