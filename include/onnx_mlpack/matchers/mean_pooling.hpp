/**
 * @file mean_pooling.hpp
 * @author Ryan Curtin
 *
 * Candidate ONNX subgraph that can match the ONNX GlobalAveragePool operation
 * to the mlpack MeanPooling layer.
 */
#ifndef ONNX_MLPACK_MATCHERS_MEAN_POOLING_HPP
#define ONNX_MLPACK_MATCHERS_MEAN_POOLING_HPP

namespace onnx_mlpack {

class MeanPoolingSubgraph : public Subgraph
{
 public:
  MeanPoolingSubgraph() : Subgraph({ "GlobalAveragePool" }) { }

  inline const char* Name() const { return "MeanPooling"; }

  inline bool Validate(const arma::uvec& indices,
                       const onnx::GraphProto& graph) const override;

  inline void Convert(const arma::uvec& indices,
                      const onnx::GraphProto& graph,
                      mlpack::DAGNetwork<>& network) const override;
};

} // namespace onnx_mlpack

#endif
