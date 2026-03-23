/**
 * @file linear_gemm.hpp
 * @author Ryan Curtin
 *
 * Candidate ONNX subgraph that can match the Linear layer with a Gemm ONNX
 * operation..
 */
#ifndef ONNX_MLPACK_MATCHERS_LINEAR_GEMM_HPP
#define ONNX_MLPACK_MATCHERS_LINEAR_GEMM_HPP

namespace onnx_mlpack {

class LinearGemmSubgraph : public Subgraph
{
 public:
  LinearGemmSubgraph() : Subgraph({ "Gemm" }) { }

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
