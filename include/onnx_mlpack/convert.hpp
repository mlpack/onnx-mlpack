/**
 * @file convert.hpp
 * @author Kumar Utkarsh
 * @author Ryan Curtin
 *
 * Top-level functions for converting ONNX graphs to mlpack DAGNetworks.
 */
#ifndef ONNX_MLPACK_CONVERT_HPP
#define ONNX_MLPACK_CONVERT_HPP

#include <onnx/onnx_pb.h>
#include <mlpack.hpp>

namespace onnx_mlpack {

/**
 * Load an ONNX model from the specified file path and perform shape inference
 * on it.
 *
 * @param filepath The path to the .onnx file.
 * @return onnx::GraphProto The graph representation of the ONNX model.
 */
inline onnx::GraphProto Load(const std::string& filePath);

/**
 * Simplify the given ONNX graph for conversion to mlpack:
 *
 *  - Unnecessary Reshapes are removed.
 *  - Unnecessary Add and Mul operators are removed.
 *  - Identity nodes are removed.
 *
 * This does in-place modification of the given graph.
 */
inline void Simplify(onnx::GraphProto& graph);

/**
 * Load and convert an ONNX model graph to an mlpack DAGNetwork by matching
 * subgraphs of the ONNX model to individual mlpack layers.
 *
 * If the ONNX model cannot be converted into an mlpack DAGNetwork, a
 * std::runtime_error will be thrown with more details.
 *
 * @param graph The ONNX model's graph representation.
 * @return mlpack::DAGNetwork<> The equivalent mlpack FFN model.
 */
inline mlpack::DAGNetwork<> Convert(const std::string& filename);

/**
 * Convert an ONNX model graph to an mlpack DAGNetwork by matching
 * subgraphs of the ONNX model to individual mlpack layers.  Make sure that
 * `Simplify()` has been called on the ONNX graph first.
 *
 * If the ONNX model cannot be converted into an mlpack DAGNetwork, a
 * std::runtime_error will be thrown with more details.
 *
 * @param graph The ONNX model's graph representation.
 * @return mlpack::DAGNetwork<> The equivalent mlpack FFN model.
 */
inline mlpack::DAGNetwork<> Convert(const onnx::GraphProto& graph);

} // namespace onnx_mlpack

#include "convert_impl.hpp"

#endif // ONNX_MLPACK_CONVERT_HPP
