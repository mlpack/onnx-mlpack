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
  if (nodes[0] > graph.node_size())
    return false;

  // Sanity check the attributes of the PReLU to ensure that we actually can do
  // the conversion.
  const onnx::NodeProto& prelu = graph.node(nodes[0]);
  if (prelu.op_type() != "PReLU")
    return false;

  // We require that the slope parameter is of size 1.
  const std::string slopeName = prelu.input(1);
  bool foundInitializer = false;
  for (size_t i = 0; i < graph.initializer_size(); ++i)
  {
    if (graph.initializer(i).has_name() &&
        graph.initializer(i).name() == slopeName)
    {
      for (size_t d = 0; d < graph.initializer(i).dims_size(); ++d)
      {
        // All dimensions must be 1 (the slope must be a scalar).
        if (graph.initializer(i).dims(d) != 1)
          return false;
      }

      break;
    }
  }

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

  double alpha = 0.0;
  for (size_t i = 0; i < graph.initializer_size(); ++i)
  {
    if (graph.initializer(i).has_name() &&
        graph.initializer(i).name() == slopeName &&
        graph.initializer(i).dims_size() == 2)
    {
      const onnx::TensorProto& t = graph.initializer(i);
      switch (t.data_type())
      {
        case onnx::TensorProto::FLOAT:
          alpha = (double) (t.has_raw_data() ?
              ((const float*) t.raw_data().data())[0] :
              t.float_data().data()[0]);
          break;

        case onnx::TensorProto::UINT8:
        case onnx::TensorProto::UINT16:
          alpha = (double) (t.has_raw_data() ?
              ((const uint32_t*) t.raw_data().data())[0] :
              t.int32_data().data()[0]);
          break;

        case onnx::TensorProto::INT8:
        case onnx::TensorProto::INT16:
        case onnx::TensorProto::INT32:
          alpha = (double) (t.has_raw_data() ?
              ((const int32_t*) t.raw_data().data())[0] :
              t.int32_data().data()[0]);
          break;

        case onnx::TensorProto::INT64:
          alpha = (double) (t.has_raw_data() ?
              ((const int64_t*) t.raw_data().data())[0] :
              t.int64_data().data()[0]);
          break;

        case onnx::TensorProto::DOUBLE:
          alpha = (t.has_raw_data() ?
              ((const double*) t.raw_data().data())[0] :
              t.double_data().data()[0]);
          break;

        case onnx::TensorProto::BOOL:
          alpha = (double) ((unsigned char*) t.raw_data().data())[0];
          break;

        case onnx::TensorProto::UINT32:
        case onnx::TensorProto::UINT64:
          alpha = (double) (t.has_raw_data() ?
              ((const uint64_t*) t.raw_data().data())[0] :
              t.uint64_data().data()[0]);
          break;

        default:
          throw std::runtime_error("PReLUSubgraph::TransferWeights(): "
              "unknown data type for PReLU slope tensor!");
      }

      l->Parameters()[0] = alpha;
      // The weight is successfully transferred, so, nothing else to do.
      return;
    }
  }

  // If we got to here, then we failed!
  throw std::runtime_error("PReLUSubgraph::TransferWeights(): "
      "failed to find slope tensor in ONNX graph!");
}

} // namespace onnx_mlpack

#endif
