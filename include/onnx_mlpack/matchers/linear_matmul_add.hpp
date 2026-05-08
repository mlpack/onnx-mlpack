/**
 * @file linear_matmul_add.hpp
 * @author Ryan Curtin
 *
 * Candidate ONNX subgraph that can match the Linear layer to a GEMM combined
 * MatMul/Add operation.
 */
#ifndef ONNX_MLPACK_MATCHERS_LINEAR_MATMUL_ADD_HPP
#define ONNX_MLPACK_MATCHERS_LINEAR_MATMUL_ADD_HPP

namespace onnx_mlpack {

class LinearMatMulAddSubgraph : public Subgraph
{
 public:
  // Required structure: MatMul -> Add.
  LinearMatMulAddSubgraph() : Subgraph({ "MatMul", "Add" }, { { 0, 1 } }) { }

  inline const char* Name() const { return "LinearMatMulAdd"; }

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
