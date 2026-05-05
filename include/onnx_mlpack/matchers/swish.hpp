/**
 * @file swish.hpp
 * @author Ryan Curtin
 *
 * Candidate ONNX subgraphs that can match the Swish layer.
 */
#ifndef ONNX_MLPACK_MATCHERS_SWISH_HPP
#define ONNX_MLPACK_MATCHERS_SWISH_HPP

namespace onnx_mlpack {

class SwishSubgraph : public Subgraph
{
 public:
  SwishSubgraph() : Subgraph({ "Swish" }) { }

  inline const char* Name() { return "Swish"; }

  inline bool Validate(const arma::uvec& indices,
                       const onnx::GraphProto& graph) const override;

  inline void Convert(const arma::uvec& indices,
                      const onnx::GraphProto& graph,
                      mlpack::DAGNetwork<>& network) const override;
};

} // namespace onnx_mlpack

#endif
