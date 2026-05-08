/**
 * @file hard_sigmoid_multi_op.hpp
 * @author Ryan Curtin
 *
 * Candidate ONNX subgraphs that can match the HardSigmoid layer when it is
 * represented as a series of operations.
 */
#ifndef ONNX_MLPACK_MATCHERS_HARD_SIGMOID_MULTI_OP_HPP
#define ONNX_MLPACK_MATCHERS_HARD_SIGMOID_MULTI_OP_HPP

namespace onnx_mlpack {

class HardSigmoidMultiOpSubgraph : public Subgraph
{
 public:
  // The operation is represented as Clip(x + a, 0, 2a) / 2a.
  // Note that mlpack only has support for a = 2.5!
  HardSigmoidMultiOpSubgraph() : Subgraph({ "Add", "Clip", "Mul" },
      { { 0, 1 }, { 1, 2 } }) { }

  inline const char* Name() const { return "HardSigmoidMultiOp"; }

  inline bool Validate(const arma::uvec& indices,
                       const onnx::GraphProto& graph) const override;

  inline void Convert(const arma::uvec& indices,
                      const onnx::GraphProto& graph,
                      mlpack::DAGNetwork<>& network) const override;
};

} // namespace onnx_mlpack

#endif
