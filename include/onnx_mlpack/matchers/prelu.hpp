/**
 * @file prelu.hpp
 * @author Ryan Curtin
 *
 * Candidate ONNX subgraphs that can match the PReLU layer.
 */
#ifndef ONNX_MLPACK_MATCHERS_PRELU_HPP
#define ONNX_MLPACK_MATCHERS_PRELU_HPP

namespace onnx_mlpack {

class PReLUSubgraph : public Subgraph
{
 public:
  PReLUSubgraph() : Subgraph({ "PRelu" }) { }

  inline const char* Name() const { return "PReLU"; }

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
