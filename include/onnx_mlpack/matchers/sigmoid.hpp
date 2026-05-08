/**
 * @file sigmoid.hpp
 * @author Ryan Curtin
 *
 * Candidate ONNX subgraphs that can match the Sigmoid layer.
 */
#ifndef ONNX_MLPACK_MATCHERS_SIGMOID_HPP
#define ONNX_MLPACK_MATCHERS_SIGMOID_HPP

namespace onnx_mlpack {

class SigmoidSubgraph : public Subgraph
{
 public:
  SigmoidSubgraph() : Subgraph({ "Sigmoid" }) { }

  inline const char* Name() const { return "Sigmoid"; }

  inline bool Validate(const arma::uvec& indices,
                       const onnx::GraphProto& graph) const override;

  inline void Convert(const arma::uvec& indices,
                      const onnx::GraphProto& graph,
                      mlpack::DAGNetwork<>& network) const override;
};

} // namespace onnx_mlpack

#endif
