/**
 * @file gemm.hpp
 * @author Kumar Utkarsh
 *
 * Handling of ONNX GEMM operation conversion.
 */
#ifndef ONNX_MLPACK_OPERATORS_GEMM_HPP
#define ONNX_MLPACK_OPERATORS_GEMM_HPP

#include "../utils.hpp"

namespace onnx_mlpack {

/**
 * The ONNX GEMM implementation performs the operation
 *
 * Y = alpha * t(A) * t(B) + beta * C
 *
 * where t(A) is either the transposed or un-transposed version of A, depending
 * on the parameters of the GEMM operator.
 *
 * In order to map this to either the mlpack Linear or LinearNoBias layer, we
 * can only handle when beta = 0.
 */

inline std::vector<size_t> AddGemm(
    mlpack::DAGNetwork<>& dag,
    onnx::GraphProto graph,
    onnx::NodeProto node,
    std::map<std::string, double> onnxOperatorAttribute);

inline void TransferWeightToGemm(
    mlpack::DAGNetwork<>& dag,
    std::vector<size_t>& layerIndex,
    onnx::GraphProto& graph,
    const onnx::NodeProto& node,
    std::map<std::string, double> onnxOperatorAttribute);

inline size_t FindOutputDimension(onnx::GraphProto graph, onnx::NodeProto node);

inline arma::mat ExtractWeights(onnx::GraphProto graph,
                                onnx::NodeProto node,
                                bool transposed);

inline arma::mat ExtractBiases(onnx::GraphProto graph, onnx::NodeProto node);

} // namespace onnx_mlpack

#include "gemm_impl.hpp"

#endif
