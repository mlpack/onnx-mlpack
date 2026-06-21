/**
 * @file conv_add.hpp
 * @author Ryan Curtin
 *
 * Candidate ONNX subgraphs that can match the Conv layer when it is followed by
 * an Add (per-feature-map bias).
 */
#ifndef ONNX_MLPACK_MATCHERS_CONV_ADD_HPP
#define ONNX_MLPACK_MATCHERS_CONV_ADD_HPP

namespace onnx_mlpack {

class ConvAddSubgraph : public Subgraph
{
 public:
  ConvAddSubgraph() : Subgraph({ "Conv", "Add" }, { { 0, 1 } }) { }

  inline const char* Name() const { return "ConvAdd"; }

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
