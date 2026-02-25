/**
 * @file convert.hpp
 * @author Kumar Utkarsh
 * @author Ryan Curtin
 *
 * Top-level functions for converting ONNX graphs to mlpack DAGNetworks.
 */
#ifndef CONVERT_HPP
#define CONVERT_HPP

#include "attribute.hpp"
#include "add_layer.hpp"

namespace onnx_mlpack {

/**
 * Load an ONNX model from the specified file path.
 *
 * This function reads an ONNX model from the provided file path and returns the
 * corresponding GraphProto object.
 *
 * @param filepath The path to the .onnx file.
 * @return onnx::GraphProto The graph representation of the ONNX model.
 */
inline onnx::GraphProto GetGraph(const std::string& filePath);

/**
 * Convert an ONNX model graph to an mlpack FFN model.
 *
 * The core logic for converting an ONNX model into an mlpack FFN model is
 * implemented in this function. Refer to the implementation for detailed steps.
 *
 * @param graph The ONNX model's graph representation.
 * @return mlpack::DAGNetwork<> The equivalent mlpack FFN model.
 */
inline mlpack::DAGNetwork<> Convert(onnx::GraphProto& graph);

} // namespace onnx_mlpack

#include "convert_impl.hpp"

#endif // CONVERT_HPP
