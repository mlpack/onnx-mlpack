/**
 * @file gelu_exact_multi_op.hpp
 * @author Ryan Curtin
 *
 * Candidate ONNX subgraphs that can match the exact GELU layer when it is
 * represented by several operations.
 */
#ifndef ONNX_MLPACK_MATCHERS_GELU_EXACT_MULTI_OP_HPP
#define ONNX_MLPACK_MATCHERS_GELU_EXACT_MULTI_OP_HPP

namespace onnx_mlpack {

class GELUExactMultiOpSubgraph : public Subgraph
{
 public:
  //
  // TensorFlow exports an exact GELU activation layer as
  //   (0.5 * x) * (Erfc(-x * sqrt(2)))
  // which uses the symmetry of Erfc().
  //
  // However, note that Erfc is not actually a valid ONNX operation!  (It's
  // called "Erf".) This appears to be a bug with TensorFlow/ONNX export,
  // mentioned off-hand in this unrelated issue:
  //
  // https://github.com/onnx/tensorflow-onnx/issues/2347
  //
  GELUExactMultiOpSubgraph() : Subgraph({ "Mul", "Neg", "Mul", "Erfc", "Mul" },
      { { 1, 2 }, // Neg -> Mul2
        { 2, 3 }, // Mul2 -> Erfc
        { 0, 4 }, { 3, 4 } }) { } // [Mul, Erfc] -> Mul

  inline const char* Name() const { return "GELUExactMultiOp"; }

  inline bool Validate(const arma::uvec& indices,
                       const onnx::GraphProto& graph) const override;

  inline void Convert(const arma::uvec& indices,
                      const onnx::GraphProto& graph,
                      mlpack::DAGNetwork<>& network) const override;
};

} // namespace onnx_mlpack

#endif
