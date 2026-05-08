/**
 * @file prelu_multi_op.hpp
 * @author Ryan Curtin
 *
 * Candidate ONNX subgraphs that can match the PReLU layer when it is
 * represented as a series of operations.  This is done by the tf2onnx tool.
 */
#ifndef ONNX_MLPACK_MATCHERS_PRELU_MULTI_OP_HPP
#define ONNX_MLPACK_MATCHERS_PRELU_MULTI_OP_HPP

namespace onnx_mlpack {

class PReLUMultiOpSubgraph : public Subgraph
{
 public:
  // TensorFlow exports a PReLU layer as:
  //     Relu(x) + (a * Relu(-x))
  PReLUMultiOpSubgraph() : Subgraph({ "Relu", "Neg", "Relu", "Mul", "Add" },
      { { 1, 2 }, // Neg -> Relu
        { 2, 3 }, // Relu -> Mul
        { 0, 4 }, { 3, 4 } }) { } // [Relu, Mul] -> Add

  inline const char* Name() const { return "PReLUMultiOp"; }

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
