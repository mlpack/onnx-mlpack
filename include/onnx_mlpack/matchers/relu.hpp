/**
 * @file relu.hpp
 * @author Ryan Curtin
 *
 * Candidate ONNX subgraphs that can match the ReLU layer.
 */
#ifndef ONNX_MLPACK_MATCHERS_RELU_HPP
#define ONNX_MLPACK_MATCHERS_RELU_HPP

namespace onnx_mlpack {

class ReLUSubgraph : public Subgraph
{
 public:
  ReLUSubgraph() : Subgraph({ "Relu" }) { }

  inline const char* Name() { return "ReLU"; }

  inline bool Validate(const arma::uvec& indices,
                       const onnx::GraphProto& graph) const override;

  inline void Convert(const arma::uvec& indices,
                      const onnx::GraphProto& graph,
                      mlpack::DAGNetwork<>& network) const override;
};

} // namespace onnx_mlpack

#endif
