/**
 * @file leaky_relu.hpp
 * @author Ryan Curtin
 *
 * Candidate ONNX subgraphs that can match the LeakyReLU layer.
 */
#ifndef ONNX_MLPACK_MATCHERS_LEAKY_RELU_HPP
#define ONNX_MLPACK_MATCHERS_LEAKY_RELU_HPP

namespace onnx_mlpack {

class LeakyReLUSubgraph : public Subgraph
{
 public:
  LeakyReLUSubgraph() : Subgraph({ "LeakyRelu" }) { }

  inline const char* Name() const { return "LeakyReLU"; }

  inline bool Validate(const arma::uvec& indices,
                       const onnx::GraphProto& graph) const override;

  inline void Convert(const arma::uvec& indices,
                      const onnx::GraphProto& graph,
                      mlpack::DAGNetwork<>& network) const override;
};

} // namespace onnx_mlpack

#endif
