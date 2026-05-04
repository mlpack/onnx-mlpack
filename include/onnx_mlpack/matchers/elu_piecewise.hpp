/**
 * @file elu_piecewise.hpp
 * @author Ryan Curtin
 *
 * Candidate ONNX subgraphs that can match the ELU layer when it is represented
 * as a strange series of piecewise operations (this is the default output of
 * TensorFlow).
 */
#ifndef ONNX_MLPACK_MATCHERS_ELU_PIECEWISE_HPP
#define ONNX_MLPACK_MATCHERS_ELU_PIECEWISE_HPP

namespace onnx_mlpack {

class ELUPiecewiseSubgraph : public Subgraph
{
 public:
  // The way TensorFlow represents an ELU in ONNX is:
  //
  // output = ((alpha * Elu(x)) * Cast(Not(x > 0), Int)) +
  //          Elu(x) * Cast(x > 0, Int)
  //
  // This is an extremely awkward and inefficient way to represent an Elu, but
  // it's what TensorFlow does, so, here we are.

  ELUPiecewiseSubgraph() : Subgraph(
      { "Greater", "Elu", "Not", "Cast", "Cast", "Mul", "Mul", "Mul", "Add" },
      { { 0, 2 }, // Not(x > 0)
        { 2, 3 }, // Cast(Not(x > 0), Int)
        { 0, 4 }, // Cast(x > 0, Int)
        { 1, 5 }, // alpha * Elu(x)
        { 5, 6 }, { 3, 6 }, // ((alpha * Elu(x)) * Cast(Not(x > 0), Int))
        { 1, 7 }, { 4, 7 }, // Elu(x) * Cast(x > 0, Int)
        { 6, 8 }, { 7, 8 } /* final addition */ }) { }

  inline const char* Name() { return "ELUPiecewise"; }

  inline bool Validate(const arma::uvec& indices,
                       const onnx::GraphProto& graph) const override;

  inline void Convert(const arma::uvec& indices,
                      const onnx::GraphProto& graph,
                      mlpack::DAGNetwork<>& network) const override;
};

} // namespace onnx_mlpack

#endif
