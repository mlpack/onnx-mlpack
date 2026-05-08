/**
 * @file softplus_threshold.hpp
 * @author Ryan Curtin
 *
 * Candidate ONNX subgraphs that can match the Softplus layer when it is
 * exported with special operations for the PyTorch threshold parameter.
 */
#ifndef ONNX_MLPACK_MATCHERS_SOFTPLUS_THRESHOLD_HPP
#define ONNX_MLPACK_MATCHERS_SOFTPLUS_THRESHOLD_HPP

namespace onnx_mlpack {

class SoftplusThresholdSubgraph : public Subgraph
{
 public:
  SoftplusThresholdSubgraph() : Subgraph(
      { "Softplus", "Greater", "Where" }, { { 0, 2 }, { 1, 2 } } ) { }

  inline const char* Name() const { return "SoftplusThreshold"; }

  inline bool Validate(const arma::uvec& indices,
                       const onnx::GraphProto& graph) const override;

  inline void Convert(const arma::uvec& indices,
                      const onnx::GraphProto& graph,
                      mlpack::DAGNetwork<>& network) const override;
};

} // namespace onnx_mlpack

#endif
