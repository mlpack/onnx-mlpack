/**
 * @file hard_sigmoid.hpp
 * @author Ryan Curtin
 *
 * Candidate ONNX subgraphs that can match the HardSwish layer using a
 * HardSigmoid ONNX node.
 */
#ifndef ONNX_MLPACK_MATCHERS_HARD_SWISH_SIGMOID_HPP
#define ONNX_MLPACK_MATCHERS_HARD_SWISH_SIGMOID_HPP

namespace onnx_mlpack {

class HardSwishSigmoidSubgraph : public Subgraph
{
 public:
  HardSwishSigmoidSubgraph() : Subgraph({ "HardSigmoid" }) { }

  inline bool Validate(const arma::uvec& indices,
                       const onnx::GraphProto& graph) const override;

  inline void Convert(const arma::uvec& indices,
                      const onnx::GraphProto& graph,
                      mlpack::DAGNetwork<>& network) const override;
};

} // namespace onnx_mlpack

#endif
