/**
 * @file softplus_threshold_impl.hpp
 * @author Ryan Curtin
 *
 * Candidate ONNX subgraphs that can match the Softplus layer when exported with
 * extra layers for thresholding.
 */
#ifndef ONNX_MLPACK_MATCHERS_SOFTPLUS_THRESHOLD_IMPL_HPP
#define ONNX_MLPACK_MATCHERS_SOFTPLUS_THRESHOLD_IMPL_HPP

#include "softplus_threshold.hpp"

namespace onnx_mlpack {

/**
 * Check that a given matching can feasibly convert to a Softplus layer.
 */
inline bool SoftplusThresholdSubgraph::Validate(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph) const
{
  if (nodes.n_elem != 3)
    return false;
  if (nodes[0] >= graph.node_size())
    return false;
  if (nodes[1] >= graph.node_size())
    return false;
  if (nodes[2] >= graph.node_size())
    return false;

  // There are no parameters to the Softplus layer, so if the name is right then
  // it is valid.
  const onnx::NodeProto& softplus = graph.node(nodes[0]);
  if (softplus.op_type() != "Softplus")
    return false;
  const onnx::NodeProto& greater = graph.node(nodes[1]);
  if (greater.op_type() != "Greater")
    return false;
  const onnx::NodeProto& where = graph.node(nodes[2]);
  if (where.op_type() != "Where")
    return false;

  // We must ensure that one of the inputs to the Where is the same as the one
  // of the inputs to the Softplus.
  if (softplus.input(0) != where.input(0) &&
      softplus.input(0) != where.input(1) &&
      softplus.input(0) != where.input(2) &&
      softplus.input(1) != where.input(0) &&
      softplus.input(1) != where.input(1) &&
      softplus.input(1) != where.input(2))
    return false;

  // We can't have any broadcasting in the Where operation.
  const std::string& whereA = where.input(0);
  const std::string& whereB = where.input(1);
  const std::string& whereC = where.input(2);
  size_t whereADims = 0;
  size_t whereBDims = 0;
  size_t whereCDims = 0;
  for (size_t i = 0; i < graph.initializer_size(); ++i)
  {
    const onnx::TensorProto& t = graph.initializer(i);
    if (t.has_name() && t.name() == whereA)
      whereADims = t.dims_size();
    else if (t.has_name() && t.name() == whereB)
      whereBDims = t.dims_size();
    else if (t.has_name() && t.name() == whereC)
      whereCDims = t.dims_size();
  }

  // Now check the ValueInfoProtos too.  Hopefully shape inference was
  // successful!
  for (size_t i = 0; i < graph.value_info_size(); ++i)
  {
    const onnx::ValueInfoProto& v = graph.value_info(i);
    if (v.has_name() && v.name() == whereA && v.has_type() &&
        v.type().has_tensor_type() && v.type().tensor_type().has_shape())
      whereADims = v.type().tensor_type().shape().dim_size();

    if (v.has_name() && v.name() == whereB && v.has_type() &&
        v.type().has_tensor_type() && v.type().tensor_type().has_shape())
      whereBDims = v.type().tensor_type().shape().dim_size();

    if (v.has_name() && v.name() == whereC && v.has_type() &&
        v.type().has_tensor_type() && v.type().tensor_type().has_shape())
      whereCDims = v.type().tensor_type().shape().dim_size();
  }

  // Make sure we found the initializers.
  if (whereADims == 0 || whereBDims == 0 || whereCDims == 0)
    return false;

  // Make sure they have the same number of dimensions; if so, we are not
  // broadcasting.
  if (whereADims != whereBDims || whereBDims != whereCDims)
    return false;

  // We have to recover the actual value used by the Greater node's scalar
  // input; if it's not the type maximum for the given type, we can't use it.
  // (The mlpack SoftPlus implementation assumes a threshold of the type
  // maximum.)
  const std::string& greaterB = greater.input(1);
  bool found = false;
  for (size_t i = 0; i < graph.initializer_size(); ++i)
  {
    const onnx::TensorProto& t = graph.initializer(i);
    if (t.has_name() && t.name() == greaterB)
    {
      if (t.dims_size() > 0)
        return false; // must be a zero-dimensional tensor

      switch (t.data_type())
      {
        case onnx::TensorProto::FLOAT:
          {
            const float x = *((float*) (t.has_raw_data() ?
                (const float*) t.raw_data().data() : t.float_data().data()));
            if (x != std::numeric_limits<float>::max())
              return false;
          }
          break;

        case onnx::TensorProto::DOUBLE:
          {
            const double x = *((double*) (t.has_raw_data() ?
                (const double*) t.raw_data().data() : t.double_data().data()));
            if (x != std::numeric_limits<double>::max())
              return false;
          }
          break;

        default: // unknown or unsupported type
          return false;
      }

      found = true;
      break;
    }
  }

  if (!found)
    return false;

  return true;
}

/**
 * Create a Softplus layer.
 */
inline void SoftplusThresholdSubgraph::Convert(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph,
    mlpack::DAGNetwork<>& network) const
{
  // We only need to add the SoftPlus layer---there is nothing else to do.
  network.Add<mlpack::SoftPlus>();
}

} // namespace onnx_mlpack

#endif
