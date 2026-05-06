/**
 * @file mish_multi_op.hpp
 * @author Ryan Curtin
 *
 * Candidate ONNX subgraphs that can match the Mish layer when it is serialized
 * as a series of ONNX operations.
 */
#ifndef ONNX_MLPACK_MATCHERS_MISH_MULTI_OP_HPP
#define ONNX_MLPACK_MATCHERS_MISH_MULTI_OP_HPP

namespace onnx_mlpack {

class MishMultiOpSubgraph : public Subgraph
{
 public:
  MishMultiOpSubgraph() : Subgraph(
      { "Softplus", "Tanh", "Mul" }, { { 0, 1 }, { 1, 2 } } ) { }

  inline const char* Name() const { return "MishMultiOp"; }

  inline bool Validate(const arma::uvec& indices,
                       const onnx::GraphProto& graph) const override;

  inline void Convert(const arma::uvec& indices,
                      const onnx::GraphProto& graph,
                      mlpack::DAGNetwork<>& network) const override;
};

} // namespace onnx_mlpack

#endif
