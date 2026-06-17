/**
 * @file apply_initial_reshapes.hpp
 * @author Ryan Curtin
 *
 * If a network has Reshape operations applied to any initialized tensors,
 * remove those Reshape operations and change the sizes of the initialized
 * tensors.
 */
#ifndef ONNX_MLPACK_APPLY_INITIAL_RESHAPES_HPP
#define ONNX_MLPACK_APPLY_INITIAL_RESHAPES_HPP

#include "tensor_to_arma.hpp"
#include "extract_attribute.hpp"

namespace onnx_mlpack {

inline void ApplyInitialReshapes(onnx::GraphProto& graph)
{
  // Iterate over all the nodes in the graph and look for Reshape operations.
  std::set<size_t> nodesToRemove;
  std::set<std::string> initializersToRemove;
  for (size_t n = 0; n < graph.node_size(); ++n)
  {
    const onnx::NodeProto& node = graph.node(n);
    if (node.op_type() != "Reshape")
      continue;

    if (node.input_size() != 2)
      continue; // Somehow the node is invalid.

    // We have a Reshape node.  Get its inputs and outputs.
    const std::string& dataName = node.input(0);
    const std::string& shapeName = node.input(1);

    size_t dataIndex = graph.initializer_size();
    size_t shapeIndex = graph.initializer_size();
    for (size_t i = 0; i < graph.initializer_size(); ++i)
    {
      if (graph.initializer(i).has_name())
      {
        if (graph.initializer(i).name() == dataName)
          dataIndex = i;
        else if (graph.initializer(i).name() == shapeName)
          shapeIndex = i;
      }
    }

    if (dataIndex == graph.initializer_size())
      continue; // We don't have the data.
    if (shapeIndex == graph.initializer_size())
      continue; // We don't have the shape.

    onnx::TensorProto& dataTensor = *graph.mutable_initializer(dataIndex);
    const onnx::TensorProto& shapeTensor = graph.initializer(shapeIndex);

    // Convert the tensor to an Armadillo object.  ONNX requires that the type
    // of the shape tensor is int64, but we can be just a little more loose here
    // and handle whatever we get.
    arma::Mat<int64_t> targetShape = TensorToArma<int64_t>(shapeTensor);
    if (targetShape.n_rows != 1 && targetShape.n_cols != 1)
      continue; // Something is wrong, the shape shouldn't be 2-dimensional.

    // Now we need to get the current shape of the data tensor.
    if (dataTensor.dims_size() == 0)
      continue; // The data has no shape at all...
    arma::Col<int64_t> dataShape(dataTensor.dims_size());
    for (size_t i = 0; i < dataTensor.dims_size(); ++i)
      dataShape[i] = (int64_t) dataTensor.dims(i);

    // Check that the target shape is actually compatible with the shape of the
    // data.
    size_t numZeroDims = 0;
    size_t numNegOneDims = 0;
    size_t numInvalidNegDims = 0;
    size_t negOneDimIndex = targetShape.n_elem;
    for (size_t i = 0; i < targetShape.n_elem; ++i)
    {
      if (targetShape[i] == 0)
      {
        ++numZeroDims;
      }
      else if (targetShape[i] == -1)
      {
        ++numNegOneDims;
        negOneDimIndex = i;
      }
      else if (targetShape[i] < -1)
      {
        ++numInvalidNegDims;
      }
    }

    // Check that the shape is actually valid.
    if (numNegOneDims > 1)
      continue;
    if (numInvalidNegDims > 0)
      continue;

    // Check whether we are actually truly allowing zero dimensions.
    int allowZero = 0;
    ExtractAttribute(node, "allowzero", allowZero);
    if (allowZero != 0 && allowZero != 1)
      continue; // Invalid allowzero setting... ignore the Reshape.

    // In this case, the number of input dimensions must exactly match the
    // number of output dimensions.
    size_t totalInputDimensions = 1;
    for (size_t i = 0; i < dataShape.n_elem; ++i)
      totalInputDimensions *= dataShape[i];

    size_t totalOutputDimensions = 1;
    bool invalid = false;
    for (size_t i = 0; i < targetShape.n_elem; ++i)
    {
      if (targetShape[i] == -1)
      {
        continue; // Ignore this dimension; it will be inferred.
      }
      else if (allowZero == 0 && targetShape[i] == 0 && i >= dataShape.n_elem)
      {
        invalid = true;
        break;
      }
      else if (allowZero == 0 && targetShape[i] == 0 && i < dataShape.n_elem)
      {
        totalOutputDimensions *= dataShape[i];
        targetShape[i] = dataShape[i];
      }
      else
      {
        totalOutputDimensions *= targetShape[i];
      }
    }

    if (invalid)
      continue; // The target shape uses dimensions the input does not have.
    if (totalInputDimensions != totalOutputDimensions)
      continue; // The target shape doesn't match the input.

    // Check that we can infer a dimension correctly if we need to.
    if (numNegOneDims == 1)
    {
      targetShape[negOneDimIndex] =
          (totalInputDimensions / totalOutputDimensions);

      // Did it divide cleanly?
      const size_t overallOutputDimensions = targetShape[negOneDimIndex] *
          totalOutputDimensions;
      if (overallOutputDimensions != totalInputDimensions)
        continue; // It did not divide cleanly; the reshape is invalid.
    }

    // Now we have the correctly inferred size.
    //
    // We can update the existing initializer shape, update the input to any
    // nodes that use the output of the reshape, and then remove the reshape
    // node.
    for (size_t i = 0; i < targetShape.n_elem; ++i)
    {
      if (i < dataTensor.dims_size())
        dataTensor.set_dims(i, targetShape[i]);
      else
        dataTensor.add_dims(targetShape[i]);
    }

    // Update any other nodes that use this reshape as output.
    const std::string& outputName = node.output(0);
    for (size_t nn = 0; nn < graph.node_size(); ++nn)
    {
      if (n == nn)
        continue; // Skip the node we are looking at.

      onnx::NodeProto& n2 = *graph.mutable_node(nn);
      for (size_t i = 0; i < n2.input_size(); ++i)
      {
        if (n2.input(i) == outputName)
          n2.set_input(i, dataTensor.name());
      }
    }

    nodesToRemove.insert(n);
    initializersToRemove.insert(shapeTensor.name());
  }

  // Now actually remove the nodes we don't need.  Protobuf only lets us remove
  // everything, then re-add it.
  for (size_t n = graph.node_size(); n > 0; --n)
    if (nodesToRemove.count(n - 1) > 0)
      graph.mutable_node()->DeleteSubrange(n - 1, 1);

  // Remove the initializers and inputs we don't need.
  for (size_t i = graph.initializer_size(); i > 0; --i)
    if (graph.initializer(i - 1).has_name() &&
        initializersToRemove.count(graph.initializer(i - 1).name()) > 0)
      graph.mutable_initializer()->DeleteSubrange(i - 1, 1);

  for (size_t i = graph.input_size(); i > 0; --i)
    if (graph.input(i - 1).has_name() &&
        initializersToRemove.count(graph.input(i - 1).name()) > 0)
      graph.mutable_input()->DeleteSubrange(i - 1, 1);
}

} // namespace onnx_mlpack

#endif
