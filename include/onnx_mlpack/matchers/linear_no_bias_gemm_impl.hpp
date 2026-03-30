/**
 * @file linear_no_bias_gemm_impl.hpp
 * @author Ryan Curtin
 *
 * Candidate ONNX subgraphs that can match the LinearNoBias layer.
 */
#ifndef ONNX_MLPACK_MATCHERS_LINEAR_NO_BIAS_GEMM_IMPL_HPP
#define ONNX_MLPACK_MATCHERS_LINEAR_NO_BIAS_GEMM_IMPL_HPP

#include "linear_no_bias_gemm.hpp"
#include "../tensor_to_arma.hpp"

namespace onnx_mlpack {

/**
 * Check that a given matching can feasibly convert to a LinearNoBias layer.
 */
inline bool LinearNoBiasGemmSubgraph::Validate(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph) const
{
  if (nodes.n_elem != 1)
    return false;
  if (nodes[0] > graph.node_size())
    return false;

  // Sanity check the attributes of the gemm to ensure that we actually can do
  // the conversion.
  const onnx::NodeProto& gemm = graph.node(nodes[0]);
  if (gemm.op_type() != "Gemm")
    return false;
  if (gemm.input_size() != 2 && gemm.input_size() != 3)
    return false;

  // We require that the second input parameter, the weights, are fully
  // initialized.
  const std::string bName = gemm.input(1);
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
    return false;

  // We cannot accept when beta is not 0 and C is specified.  (beta > 0 implies
  // recurrence!)
  if (gemm.input_size() == 3)
  {
    // TODO: use onnxOperatorAttribute or similar
    double beta = 1.0; // default according to ONNX spec
    for (size_t i = 0; i < gemm.attribute_size(); ++i)
    {
      if (gemm.attribute(i).has_name() && gemm.attribute(i).name() == "beta" &&
          gemm.attribute(i).has_f())
      {
        beta = (double) gemm.attribute(i).f();
        break;
      }
    }

    // TODO: add a little logging here?
    if (beta != 0.0)
      return false;
  }

  // The alpha attribute must be set to 1; the LinearNoBias layer doesn't
  // support constant scaling.
  double alpha = 1.0;
  for (size_t i = 0; i < gemm.attribute_size(); ++i)
  {
    if (gemm.attribute(i).has_name() && gemm.attribute(i).name() == "alpha" &&
        gemm.attribute(i).has_f())
    {
      alpha = (double) gemm.attribute(i).f();
      break;
    }
  }

  if (alpha != 1.0)
    return false;

  return true;
}

/**
 * Create a LinearNoBias layer with the same metadata as the given ONNX graph.
 */
inline void LinearNoBiasGemmSubgraph::Convert(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph,
    mlpack::DAGNetwork<>& network) const
{
  // Since there is only one ONNX node, a Gemm, we don't have anything to do
  // other than create the LinearNoBias layer.  However, we must first compute
  // the number of output nodes using the shape of the graph.
  //
  // There are a few possibilities: if C is specified, we can steal the size
  // from there.  If C is not specified, then we must infer the size based on
  // the shapes of A and B (and the settings of transA and transB).

  size_t outputDims = 0;
  const onnx::NodeProto& gemm = graph.node(nodes[0]);
  if (gemm.input_size() == 3)
  {
    const std::string cName = gemm.input(2);
    for (size_t i = 0; i < graph.initializer_size(); ++i)
    {
      if (graph.initializer(i).name() == cName &&
          graph.initializer(i).dims_size() > 0 &&
          graph.initializer(i).dims(0) > 0)
      {
        outputDims = (size_t) graph.initializer(i).dims(0);
      }
    }
  }

  // If we didn't get it directly from C, then get it from B.
  if (outputDims == 0)
  {
    // The second dimension of B is the output size.
    const std::string bName = gemm.input(1);
    size_t transB = 0;
    for (size_t i = 0; i < gemm.attribute_size(); ++i)
    {
      if (gemm.attribute(i).has_name() &&
          gemm.attribute(i).name() == "transB" &&
          gemm.attribute(i).has_i())
      {
        transB = (size_t) gemm.attribute(i).i();
        break;
      }
    }

    for (size_t i = 0; i < graph.initializer_size(); ++i)
    {
      if (graph.initializer(i).has_name() &&
          graph.initializer(i).name() == bName &&
          graph.initializer(i).dims_size() == 2)
      {
        outputDims = (transB == 0) ? graph.initializer(i).dims(1) :
            graph.initializer(i).dims(0);
      }
    }
  }

  if (outputDims == 0)
  {
    throw std::runtime_error("LinearNoBiasGemmSubgraph::Convert(): cannot "
        "infer output size of ONNX Gemm operation!");
  }

  network.Add<mlpack::LinearNoBias>(outputDims);
}

inline void LinearNoBiasGemmSubgraph::TransferWeights(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph,
    mlpack::Layer<>* layer) const
{
  // We have already concluded that the weights of the operation must be the B
  // matrix to the Gemm operation.  Therefore, we simply need to get its
  // weights.
  const onnx::NodeProto& gemm = graph.node(nodes[0]);
  const std::string bName = gemm.input(1);
  size_t transB = 0;
  for (size_t i = 0; i < gemm.attribute_size(); ++i)
  {
    if (gemm.attribute(i).has_name() &&
        gemm.attribute(i).name() == "transB" &&
        gemm.attribute(i).has_i())
    {
      transB = (size_t) gemm.attribute(i).i();
      break;
    }
  }

  mlpack::LinearNoBias<>* l = dynamic_cast<mlpack::LinearNoBias<>*>(layer);
  for (size_t i = 0; i < graph.initializer_size(); ++i)
  {
    if (graph.initializer(i).has_name() &&
        graph.initializer(i).name() == bName &&
        graph.initializer(i).dims_size() == 2)
    {
      if (transB == 1)
        l->Parameters() = TensorToArma(graph.initializer(i)).t();
      else
        l->Parameters() = TensorToArma(graph.initializer(i));

      // The weight is successfully transferred, so, nothing else to do.
      return;
    }
  }

  // If we got to here, then we failed!
  throw std::runtime_error("LinearNoBiasGemmSubgraph::TransferWeights(): "
      "failed to find weight tensor in ONNX graph!");
}

} // namespace onnx_mlpack

#endif
