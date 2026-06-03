/**
 * @file max_pooling.hpp
 * @author Ryan Curtin
 *
 * Candidate ONNX subgraphs that can match the MaxPooling layer.
 */
#ifndef ONNX_MLPACK_MATCHERS_MAX_POOLING_HPP
#define ONNX_MLPACK_MATCHERS_MAX_POOLING_HPP

namespace onnx_mlpack {

class MaxPoolingSubgraph : public Subgraph
{
 public:
  MaxPoolingSubgraph() : Subgraph({ "MaxPool" }) { }

  inline const char* Name() const { return "MaxPooling"; }

  inline bool Validate(const arma::uvec& indices,
                       const onnx::GraphProto& graph) const override;

  inline void Convert(const arma::uvec& indices,
                      const onnx::GraphProto& graph,
                      mlpack::DAGNetwork<>& network) const override;
};

} // namespace onnx_mlpack

#endif
