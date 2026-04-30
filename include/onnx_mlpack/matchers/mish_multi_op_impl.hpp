/**
 * @file mish_multi_op_impl.hpp
 * @author Ryan Curtin
 *
 * Candidate ONNX subgraphs that can match the Mish layer as a series of ONNX
 * operations.
 */
#ifndef ONNX_MLPACK_MATCHERS_MISH_MULTI_OP_IMPL_HPP
#define ONNX_MLPACK_MATCHERS_MISH_MULTI_OP_IMPL_HPP

#include "mish_multi_op.hpp"

namespace onnx_mlpack {

/**
 * Check that a given matching can feasibly convert to a ReLU layer.
 */
inline bool MishMultiOpSubgraph::Validate(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph) const
{
  std::cout << "validating MishMultiOpSubgraph\n";
  if (nodes.n_elem != 3)
    return false;
  if (nodes[0] >= graph.node_size())
    return false;
  if (nodes[1] >= graph.node_size())
    return false;
  if (nodes[2] >= graph.node_size())
    return false;
  std::cout << "valid check 1\n";

  const onnx::NodeProto& softplus = graph.node(nodes[0]);
  if (softplus.op_type() != "Softplus")
    return false;
  std::cout << "valid check 2\n";
  const onnx::NodeProto& tanh = graph.node(nodes[1]);
  if (tanh.op_type() != "Tanh")
    return false;
  std::cout << "valid check 3\n";
  const onnx::NodeProto& mul = graph.node(nodes[2]);
  if (mul.op_type() != "Mul")
    return false;
  std::cout << "valid check 4\n";

  // We must ensure that the input to the Softplus is also one of the inputs to
  // the Mul.
  if (softplus.input(0) != mul.input(0) && softplus.input(0) != mul.input(1))
    return false;
  std::cout << "valid check 5\n";

  // We cannot have broadcasting in the mul operation, so we need to ensure that
  // both inputs have the same dimensions.
  const std::string& mulA = mul.input(0);
  const std::string& mulB = mul.input(1);
  size_t mulADims = 0;
  size_t mulBDims = 0;
  for (size_t i = 0; i < graph.initializer_size(); ++i)
  {
    const onnx::TensorProto& t = graph.initializer(i);
    if (t.has_name() && t.name() == mulA)
      mulADims = t.dims_size();
    if (t.has_name() && t.name() == mulB)
      mulBDims = t.dims_size();
  }
  std::cout << "mulADims " << mulADims << " mulBDims " << mulBDims << "\n";

  // Now check the ValueInfoProtos too.  Hopefully shape inference was
  // successful!
  for (size_t i = 0; i < graph.value_info_size(); ++i)
  {
    const onnx::ValueInfoProto& v = graph.value_info(i);
    if (v.has_name() && v.name() == mulA &&
        v.has_type() && v.type().has_tensor_type() &&
        v.type().tensor_type().has_shape())
    {
      mulADims = v.type().tensor_type().shape().dim_size();
    }

    if (v.has_name() && v.name() == mulB &&
        v.has_type() && v.type().has_tensor_type() &&
        v.type().tensor_type().has_shape())
    {
      mulBDims = v.type().tensor_type().shape().dim_size();
    }
  }
  std::cout << "second mulADims " << mulADims << " mulBDims " << mulBDims << "\n";

  // Make sure we found the initializers.
  if (mulADims == 0 || mulBDims == 0)
    return false;
  std::cout << "valid check 6\n";

  // Make sure they have the same number of dimensions: if so, we are not
  // broadcasting.
  if (mulADims != mulBDims)
    return false;

  std::cout << "valid!\n";
  return true;
}

/**
 * Create a Mish layer.
 */
inline void MishMultiOpSubgraph::Convert(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph,
    mlpack::DAGNetwork<>& network) const
{
  // We only need to add the Mish layer---there is nothing else to do.
  network.Add<mlpack::Mish>();
}

} // namespace onnx_mlpack

#endif
