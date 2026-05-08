/**
 * @file linear_no_bias_gemm.hpp
 * @author Ryan Curtin
 *
 * Candidate ONNX subgraphs that can match the LinearNoBias layer.
 */
#ifndef ONNX_MLPACK_MATCHERS_LINEAR_NO_BIAS_GEMM_HPP
#define ONNX_MLPACK_MATCHERS_LINEAR_NO_BIAS_GEMM_HPP

namespace onnx_mlpack {

class LinearNoBiasGemmSubgraph : public Subgraph
{
 public:
  LinearNoBiasGemmSubgraph() : Subgraph({ "Gemm" }) { }

  inline const char* Name() const { return "LinearNoBiasGEMM"; }

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
