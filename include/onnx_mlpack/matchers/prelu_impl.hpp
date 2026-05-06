/**
 * @file prelu_impl.hpp
 * @author Ryan Curtin
 *
 * Candidate ONNX subgraphs that can match the PReLU layer.
 */
#ifndef ONNX_MLPACK_MATCHERS_PRELU_IMPL_HPP
#define ONNX_MLPACK_MATCHERS_PRELU_IMPL_HPP

#include "prelu.hpp"

namespace onnx_mlpack {

/**
 * Check that a given matching can feasibly convert to a LinearNoBias layer.
 */
inline bool PReLUSubgraph::Validate(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph) const
{
  if (nodes.n_elem != 1)
    return false;
  if (nodes[0] >= graph.node_size())
    return false;

  // Sanity check the attributes of the PReLU to ensure that we actually can do
  // the conversion.
  const onnx::NodeProto& prelu = graph.node(nodes[0]);
  if (prelu.op_type() != "PReLU")
    return false;

  // We require that the slope parameter is of size 1.
  double slope;
  if (!ExtractScalar(graph, prelu.input(1), slope))
    return false;

  return true;
}

/**
 * Create a PReLU layer with the same metadata as the given ONNX graph.
 */
inline void PReLUSubgraph::Convert(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph,
    mlpack::DAGNetwork<>& network) const
{
  // Nothing to do: we will extract the value of the slope in TransferWeights().
  network.Add<mlpack::PReLU>();
}

inline void PReLUSubgraph::TransferWeights(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph,
    mlpack::Layer<>* layer) const
{
  // We have already concluded that the weights of the operation must be the B
  // matrix to the MatMul operation.  Therefore, we simply need to get its
  // weights.
  const onnx::NodeProto& prelu = graph.node(nodes[0]);
  const std::string slopeName = prelu.input(1);
  mlpack::PReLU<>* l = dynamic_cast<mlpack::PReLU<>*>(layer);

  double alpha = DBL_MAX;
  if (!ExtractScalar(graph, slopeName, alpha))
  {
    throw std::runtime_error("PReLUSubgraph::TransferWeights(): failed to "
        "extract scalar value from slope tensor!");
  }

  l->Parameters()[0] = alpha;
}

} // namespace onnx_mlpack

#endif
