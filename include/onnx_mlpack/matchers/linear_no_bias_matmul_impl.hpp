/**
 * @file linear_no_bias_gemm_impl.hpp
 * @author Ryan Curtin
 *
 * Candidate ONNX subgraphs that can match the LinearNoBias layer.
 */
#ifndef ONNX_MLPACK_MATCHERS_LINEAR_NO_BIAS_MATMUL_IMPL_HPP
#define ONNX_MLPACK_MATCHERS_LINEAR_NO_BIAS_MATMUL_IMPL_HPP

#include "linear_no_bias_matmul.hpp"
#include "../tensor_to_arma.hpp"

namespace onnx_mlpack {

/**
 * Check that a given matching can feasibly convert to a LinearNoBias layer.
 */
inline bool LinearNoBiasMatMulSubgraph::Validate(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph) const
{
  if (nodes.n_elem != 1)
    return false;
  if (nodes[0] >= graph.node_size())
    return false;

  // Sanity check the attributes of the MatMul to ensure that we actually can do
  // the conversion.
  const onnx::NodeProto& matmul = graph.node(nodes[0]);
  if (matmul.op_type() != "MatMul")
    return false;

  // We require that the second input parameter, the weights, are fully
  // initialized.
  const std::string bName = matmul.input(1);
  bool foundInitializer = false;
  for (size_t i = 0; i < graph.initializer_size(); ++i)
  {
    if (graph.initializer(i).has_name() &&
        graph.initializer(i).name() == bName &&
        graph.initializer(i).dims_size() == 2)
    {
      foundInitializer = true;
      break;
    }
  }

  // The second input must be an input to graph---we can't accept that!
  if (!foundInitializer)
  {
    return false;
  }

  return true;
}

/**
 * Create a LinearNoBias layer with the same metadata as the given ONNX graph.
 */
inline void LinearNoBiasMatMulSubgraph::Convert(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph,
    mlpack::DAGNetwork<>& network) const
{
  // Since there is only one ONNX node, a MatMul, we don't have anything to do
  // other than create the LinearNoBias layer.  However, we must first compute
  // the number of output nodes using the shape of the graph.
  //
  // There are a few possibilities: if C is specified, we can steal the size
  // from there.  If C is not specified, then we must infer the size based on
  // the shapes of A and B (and the settings of transA and transB).

  size_t outputDims = 0;
  const onnx::NodeProto& matmul = graph.node(nodes[0]);

  // The second dimension of B is the output size.
  const std::string bName = matmul.input(1);
  for (size_t i = 0; i < graph.initializer_size(); ++i)
  {
    if (graph.initializer(i).has_name() &&
        graph.initializer(i).name() == bName &&
        graph.initializer(i).dims_size() == 2)
    {
      outputDims = graph.initializer(i).dims(1);
    }
  }

  if (outputDims == 0)
  {
    throw std::runtime_error("LinearNoBiasMatMulSubgraph::Convert(): cannot "
        "infer output size of ONNX MatMul operation!");
  }

  network.Add<mlpack::LinearNoBias>(outputDims);
}

inline void LinearNoBiasMatMulSubgraph::TransferWeights(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph,
    mlpack::Layer<>* layer) const
{
  // We have already concluded that the weights of the operation must be the B
  // matrix to the MatMul operation.  Therefore, we simply need to get its
  // weights.
  const onnx::NodeProto& matmul = graph.node(nodes[0]);
  const std::string bName = matmul.input(1);
  mlpack::LinearNoBias<>* l = dynamic_cast<mlpack::LinearNoBias<>*>(layer);

  for (size_t i = 0; i < graph.initializer_size(); ++i)
  {
    if (graph.initializer(i).has_name() &&
        graph.initializer(i).name() == bName &&
        graph.initializer(i).dims_size() == 2)
    {
      l->Parameters() = TensorToArma(graph.initializer(i));
      // The weight is successfully transferred, so, nothing else to do.
      return;
    }
  }

  // If we got to here, then we failed!
  throw std::runtime_error("LinearNoBiasMatMulSubgraph::TransferWeights(): "
      "failed to find weight tensor in ONNX graph!");
}

} // namespace onnx_mlpack

#endif
