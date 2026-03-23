/**
 * @file linear_no_bias_matmul.hpp
 * @author Ryan Curtin
 *
 * Candidate ONNX subgraphs that can match the LinearNoBias layer.
 */
#ifndef ONNX_MLPACK_MATCHERS_LINEAR_NO_BIAS_MATMUL_HPP
#define ONNX_MLPACK_MATCHERS_LINEAR_NO_BIAS_MATMUL_HPP

namespace onnx_mlpack {

class LinearNoBiasMatMulSubgraph : public Subgraph
{
 public:
  LinearNoBiasMatMulSubgraph() : Subgraph({ "MatMul" }) { }

  inline bool Validate(const arma::uvec& indices,
                       const onnx::GraphProto& graph) const override;

  inline void Convert(const arma::uvec& indices,
                      const onnx::GraphProto& graph,
                      mlpack::DAGNetwork<>& network) const override;

  inline void TransferWeights(const arma::uvec& indices,
                              const onnx::GraphProto& graph,
                              mlpack::Layer<>* layer) const override;
};

} // namespace onnx_mlpack

#endif
