/**
 * @file mul.hpp
 * @author Ryan Curtin
 *
 * Candidate ONNX subgraphs that can match a plain Mul node that multiplies a
 * scalar.
 */
#ifndef ONNX_MLPACK_MATCHERS_MUL_SCALAR_HPP
#define ONNX_MLPACK_MATCHERS_MUL_SCALAR_HPP

namespace onnx_mlpack {

class MulScalarSubgraph : public Subgraph
{
 public:
  MulScalarSubgraph() : Subgraph({ "Mul" }) { }

  inline const char* Name() const { return "MulScalar"; }

  inline bool Validate(const arma::uvec& indices,
                       const onnx::GraphProto& graph) const override;

  inline void Convert(const arma::uvec& indices,
                      const onnx::GraphProto& graph,
                      mlpack::DAGNetwork<>& network) const override;
};

} // namespace onnx_mlpack

#endif
