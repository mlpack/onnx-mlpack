/**
 * @file linear_no_bias.hpp
 * @author Ryan Curtin
 *
 * Candidate ONNX subgraphs that can match the LinearNoBias layer.
 */
#ifndef ONNX_MLPACK_MATCHERS_LINEAR_NO_BIAS_HPP
#define ONNX_MLPACK_MATCHERS_LINEAR_NO_BIAS_HPP

namespace onnx_mlpack {

class LinearNoBiasSubgraph : public Subgraph
{
 public:
  LinearNoBiasSubgraph() : Subgraph() { }
  LinearNoBiasSubgraph(const std::vector<std::string>& vertices) :
      Subgraph(vertices) { }

  inline bool Validate(const arma::uvec& indices,
                       const onnx::GraphProto& graph) const override;

  inline void Convert(const arma::uvec& indices,
                      const onnx::GraphProto& graph,
                      mlpack::DAGNetwork<>& network) const override;

  inline void TransferWeights(const arma::uvec& indices,
                              const onnx::GraphProto& graph,
                              mlpack::Layer<>* layer) const override;

  static inline std::vector<Subgraph*> Subgraphs();
};

} // namespace onnx_mlpack

#endif
