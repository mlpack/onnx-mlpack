/**
 * @file gelu_multi_op.hpp
 * @author Ryan Curtin
 *
 * Candidate ONNX subgraphs that can match the approximate GELU layer when it is
 * represented by several operations.
 */
#ifndef ONNX_MLPACK_MATCHERS_GELU_MULTI_OP_HPP
#define ONNX_MLPACK_MATCHERS_GELU_MULTI_OP_HPP

namespace onnx_mlpack {

class GELUMultiOpSubgraph : public Subgraph
{
 public:
  //
  // TensorFlow exports an approximate GELU activation layer as
  //    0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.44715 * pow(x, 3)))).
  //
  GELUMultiOpSubgraph() : Subgraph(
      { "Mul", "Pow", "Mul", "Add", "Mul", "Tanh", "Add", "Mul" },
      { { 1, 2 }, // Pow -> Mul
        { 2, 3 }, // Mul -> Add
        { 3, 4 }, // Add -> Mul
        { 4, 5 }, // Mul -> Tanh
        { 5, 6 }, // Tanh -> Add
        { 0, 7 }, { 6, 7 } }) { } // [Mul, Add] -> Mul

  inline const char* Name() const { return "GELUMultiOp"; }

  inline bool Validate(const arma::uvec& indices,
                       const onnx::GraphProto& graph) const override;

  inline void Convert(const arma::uvec& indices,
                      const onnx::GraphProto& graph,
                      mlpack::DAGNetwork<>& network) const override;
};

} // namespace onnx_mlpack

#endif
