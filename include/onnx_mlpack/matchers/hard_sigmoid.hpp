/**
 * @file hard_sigmoid.hpp
 * @author Ryan Curtin
 *
 * Candidate ONNX subgraphs that can match the HardSigmoid layer.
 */
#ifndef ONNX_MLPACK_MATCHERS_HARD_SIGMOID_HPP
#define ONNX_MLPACK_MATCHERS_HARD_SIGMOID_HPP

namespace onnx_mlpack {

class HardSigmoidSubgraph : public Subgraph
{
 public:
  HardSigmoidSubgraph() : Subgraph({ "HardSigmoid" }) { }

  inline bool Validate(const arma::uvec& indices,
                       const onnx::GraphProto& graph) const override;

  inline void Convert(const arma::uvec& indices,
                      const onnx::GraphProto& graph,
                      mlpack::DAGNetwork<>& network) const override;
};

} // namespace onnx_mlpack

#endif
