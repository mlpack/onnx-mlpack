/**
 * @file conv.hpp
 * @author Ryan Curtin
 *
 * Candidate ONNX subgraphs that can match the Conv layer.
 */
#ifndef ONNX_MLPACK_MATCHERS_CONV_HPP
#define ONNX_MLPACK_MATCHERS_CONV_HPP

namespace onnx_mlpack {

class ConvSubgraph : public Subgraph
{
 public:
  ConvSubgraph() : Subgraph({ "Conv" }) { }

  inline const char* Name() const { return "Conv"; }

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
