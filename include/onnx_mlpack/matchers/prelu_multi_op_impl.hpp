/**
 * @file prelu_multi_op_impl.hpp
 * @author Ryan Curtin
 *
 * Candidate ONNX subgraphs that can match the PReLU layer.
 */
#ifndef ONNX_MLPACK_MATCHERS_PRELU_MULTI_OP_IMPL_HPP
#define ONNX_MLPACK_MATCHERS_PRELU_MULTI_OP_IMPL_HPP

#include "prelu_multi_op.hpp"

namespace onnx_mlpack {

/**
 * Check that a given matching can feasibly convert to a LinearNoBias layer.
 */
inline bool PReLUMultiOpSubgraph::Validate(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph) const
{
  if (nodes.n_elem != 5)
    return false;
  for (size_t i = 0; i < nodes.n_elem; ++i)
    if (nodes[i] >= graph.node_size())
      return false;

  // Sanity check the attributes of the PReLU to ensure that we actually can do
  // the conversion.
  const onnx::NodeProto& reluPos = graph.node(nodes[0]);
  const onnx::NodeProto& neg = graph.node(nodes[1]);
  const onnx::NodeProto& reluNeg = graph.node(nodes[2]);
  const onnx::NodeProto& alphaMul = graph.node(nodes[3]);
  const onnx::NodeProto& finalAdd = graph.node(nodes[4]);
  if (reluPos.op_type() != "Relu" ||
      neg.op_type() != "Neg" ||
      reluNeg.op_type() != "Relu" ||
      alphaMul.op_type() != "Mul" ||
      finalAdd.op_type() != "Add")
    return false;

  // Extract the slope parameter from the Mul layer.
  double slope = DBL_MAX;
  if (!ExtractEitherScalar(graph, alphaMul.input(0), alphaMul.input(1), slope))
    return false;

  return true;
}

/**
 * Create a PReLU layer with the same metadata as the given ONNX graph.
 */
inline void PReLUMultiOpSubgraph::Convert(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph,
    mlpack::DAGNetwork<>& network) const
{
  // Nothing to do: we will extract the value of the slope in TransferWeights().
  network.Add<mlpack::PReLU>();
}

inline void PReLUMultiOpSubgraph::TransferWeights(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph,
    mlpack::Layer<>* layer) const
{
  // We have already concluded that the weights of the operation must be the B
  // matrix to the MatMul operation.  Therefore, we simply need to get its
  // weights.
  const onnx::NodeProto& alphaMul = graph.node(nodes[3]);
  double alpha = DBL_MAX;
  if (!ExtractEitherScalar(graph, alphaMul.input(0), alphaMul.input(1), alpha))
  {
    throw std::runtime_error("PReLUMultiOpSubgraph::TransferWeights(): failed "
        "to extract scalar value from slope tensor!");
  }

  mlpack::PReLU<>* l = dynamic_cast<mlpack::PReLU<>*>(layer);

  // Since the operation represented by ONNX is alpha * Relu(-x), to be correct,
  // we have to invert alpha.
  l->Parameters()[0] = -alpha;
}

} // namespace onnx_mlpack

#endif
