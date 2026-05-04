/**
 * @file linear_gemm_impl.hpp
 * @author Ryan Curtin
 *
 * Candidate ONNX subgraphs that can match the Linear layer.
 */
#ifndef ONNX_MLPACK_MATCHERS_LINEAR_GEMM_IMPL_HPP
#define ONNX_MLPACK_MATCHERS_LINEAR_GEMM_IMPL_HPP

#include "linear_gemm.hpp"
#include "../tensor_to_arma.hpp"

namespace onnx_mlpack {

/**
 * Check that a given matching can feasibly convert to a Linear layer.
 */
inline bool LinearGemmSubgraph::Validate(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph) const
{
  if (nodes.n_elem != 1)
    return false;
  if (nodes[0] >= graph.node_size())
    return false;

  // Sanity check the attributes of the gemm to ensure that we actually can do
  // the conversion.
  const onnx::NodeProto& gemm = graph.node(nodes[0]);
  if (gemm.op_type() != "Gemm")
    return false;
  if (gemm.input_size() != 3) // we must use C for this matcher.
    return false;

  // We cannot accept when beta is 0: then there is no bias.
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
  if (beta == 0.0)
    return false;

  // We require that the second and third input parameters (weights and biases)
  // are fully initialized.
  const std::string bName = gemm.input(1);
  const std::string cName = gemm.input(2);
  bool foundInitializerB = false;
  bool foundInitializerC = false;
  for (size_t i = 0; i < graph.initializer_size(); ++i)
  {
    if (graph.initializer(i).has_name() &&
        graph.initializer(i).name() == bName &&
        graph.initializer(i).dims_size() == 2)
    {
      foundInitializerB = true;
    }

    if (graph.initializer(i).has_name() &&
        graph.initializer(i).name() == cName &&
        graph.initializer(i).dims_size() == 1)
    {
      foundInitializerC = true;
    }
  }

  if (!foundInitializerB || !foundInitializerC)
    return false;

  // The alpha attribute must be set to 1; the Linear layer doesn't support
  // constant scaling.
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
 * Create a Linear layer with the same metadata as the given ONNX graph.
 */
inline void LinearGemmSubgraph::Convert(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph,
    mlpack::DAGNetwork<>& network) const
{
  // Since there is only one ONNX node, a Gemm, we don't have anything to do
  // other than create the Linear layer.  However, we must first compute the
  // number of output nodes using the shape of the graph.  We can do this by
  // taking the size of C (the biases).

  size_t outputDims = 0;
  const onnx::NodeProto& gemm = graph.node(nodes[0]);
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

  if (outputDims == 0)
  {
    throw std::runtime_error("LinearGemmSubgraph::Convert(): cannot "
        "infer output size of ONNX Gemm operation!");
  }

  network.Add<mlpack::Linear>(outputDims);
}

inline void LinearGemmSubgraph::TransferWeights(
    const arma::uvec& nodes,
    const onnx::GraphProto& graph,
    mlpack::Layer<>* layer) const
{
  // We have already concluded that the weights of the operation must be the B
  // matrix to the Gemm operation, and the biases must be the C matrix.
  const onnx::NodeProto& gemm = graph.node(nodes[0]);
  const std::string bName = gemm.input(1);
  const std::string cName = gemm.input(2);
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

  bool weightsDone = false;
  bool biasesDone = false;
  mlpack::Linear<>* l = dynamic_cast<mlpack::Linear<>*>(layer);
  for (size_t i = 0; i < graph.initializer_size(); ++i)
  {
    if (graph.initializer(i).has_name() &&
        graph.initializer(i).name() == bName &&
        graph.initializer(i).dims_size() == 2)
    {
      if (transB == 1)
        l->Weight() = TensorToArma(graph.initializer(i)).t();
      else
        l->Weight() = TensorToArma(graph.initializer(i));
      weightsDone = true;
    }

    if (graph.initializer(i).has_name() &&
        graph.initializer(i).name() == cName &&
        graph.initializer(i).dims_size() == 1)
    {
      l->Bias() = TensorToArma(graph.initializer(i)).t();
      biasesDone = true;
    }
  }

  if (!weightsDone || !biasesDone)
  {
    throw std::runtime_error("LinearGemmSubgraph::TransferWeights(): "
        "failed to find weight tensor in ONNX graph!");
  }
}

} // namespace onnx_mlpack

#endif
