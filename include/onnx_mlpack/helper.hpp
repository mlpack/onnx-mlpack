/**
 * @file helper.hpp
 * @author Kumar Utkarsh
 *
 * Utility helper functions for onnx-mlpack conversions.
 */
#ifndef ONNX_MLPACK_HELPER_HPP
#define ONNX_MLPACK_HELPER_HPP

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <queue>
#include <stack>
#include <onnx/onnx_pb.h>

namespace get {

/**
 * Retrieve the name of the input node from an ONNX graph.  This input name is
 * crucial as it allows us to fetch the dimensions of the input features. For
 * example, in a YOLO model, the input name might be "image1", which will later
 * be used in `InputDimension` to obtain the dimensions of the input feature.
 *
 * @param graph The ONNX graph.
 * @return The name of the model input.
 */
inline std::string ModelInput(onnx::GraphProto& graph);

/**
 * Get the dimensions of the input features using the name obtained from
 * `ModelInput`.
 *
 * @param graph The ONNX graph.
 * @param modelInput The name of the model input obtained from `ModelInput`.
 * @return A vector containing the dimensions {channels, height, width}.
 */
inline std::vector<size_t> InputDimension(onnx::GraphProto& graph,
                                          const std::string& modelInput);

// DependentNodes, AdjacencyMatrix, TopologicallySortedNodes, dfs
// these methods are made to get the node in topological order

inline std::vector<size_t> DependentNodes(onnx::GraphProto& graph,
                                          const std::string& nodeInput);

inline std::vector<std::vector<size_t>> AdjacencyMatrix(
    onnx::GraphProto& graph);

inline std::vector<size_t> TopologicallySortedNodes(
    onnx::GraphProto& graph,
    std::vector<std::vector<size_t>>& adj);

inline void dfs(size_t node,
                std::vector<size_t>& visited,
                const std::vector<std::vector<size_t>>& adj,
                std::stack<size_t>& st);

/**
 * Retrieve the initializer from the ONNX graph by its name.
 *
 * @param graph The ONNX graph.
 * @param initializerName The name of the initializer.
 * @return A reference to the ONNX TensorProto representing the initializer.
 */
inline const onnx::TensorProto& Initializer(
    onnx::GraphProto& graph,
    const std::string& initializerName);

/**
 * Convert ONNX initializer data from row-major to column-major format.
 *
 * ONNX stores weight data in row-major format, but mlpack expects the data in
 * column-major format.  This function handles the conversion.
 *
 * @param initializer Reference to the ONNX initializer.
 * @return The converted matrix in column-major format.
 */
inline arma::fmat ConvertToColumnMajor(const onnx::TensorProto& initializer);

/**
 * Convert an mlpack column-major matrix into row-major format.
 *
 * This utility function is useful for comparing mlpack layer outputs with ONNX
 * runtime outputs to ensure successful weight transfer.
 *
 * @param matrix The mlpack matrix in column-major format.
 * @param outputDimension The desired output dimensions {width, height,
 *     channels}.
 * @return A vector containing the data in row-major format.
 */
inline std::vector<double> ConvertToRowMajor(
    const arma::mat& matrix,
    const std::vector<size_t>& outputDimension);

} // namespace get

#include "helper_impl.hpp"

#endif // ONNX_MLPACK_HELPER_HPP
