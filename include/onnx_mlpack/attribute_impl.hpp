/**
 * @file attribute_impl.hpp
 * @author Kumar Utkarsh
 *
 * Implementation of ONNX attribute extraction.
 */
#ifndef ONNX_MLPACK_ATTRIBUTE_IMPL_HPP
#define ONNX_MLPACK_ATTRIBUTE_IMPL_HPP

#include "attribute.hpp"

inline std::map<std::string, double> OnnxOperatorAttribute(
    onnx::GraphProto& graph, const onnx::NodeProto& node)
{
  // Define default values for ONNX node attributes based on operator types.  If
  // the node does not specify certain attributes, these default values will be
  // used.
  //
  // https://github.com/Talmaj/onnx2pytorch/blob/master/onnx2pytorch/convert/attribute.py
  map<string, double> onnxOperatorAttribute;
  if (node.op_type() == "Gemm")
  {
    onnxOperatorAttribute["alpha"] = 1;
    onnxOperatorAttribute["beta"] = 1;
    onnxOperatorAttribute["transA"] = 0;
    onnxOperatorAttribute["transB"] = 0;
  }
  else if (node.op_type() == "Relu")
  {
    // TODO
  }
  else if (node.op_type() == "LeakyRelu")
  {
    onnxOperatorAttribute["alpha"] = 0.01;
  }
  else if (node.op_type() == "Softmax")
  {
    // TODO
  }
  else if (node.op_type() == "Mul")
  {
    // TODO
  }
  else if (node.op_type() == "Add")
  {
    // TODO
  }
  else if (node.op_type() == "Conv")
  {
    // Either there will be auto_pad or pads, but we will set default value for
    // both of them.

    onnxOperatorAttribute["paddingType"] = 0; // none 0, valid 1, same 2
    // NOTSET 0, SAME_UPPER 1, SAME_LOWER 2, VALID 3
    onnxOperatorAttribute["auto_pad"] = 0;
    onnxOperatorAttribute["pad_top"] = 0;
    onnxOperatorAttribute["pad_bottom"] = 0;
    onnxOperatorAttribute["pad_right"] = 0;
    onnxOperatorAttribute["pad_left"] = 0;
    onnxOperatorAttribute["group"] = 1;
    onnxOperatorAttribute["dilation_height"] = 1;
    onnxOperatorAttribute["dilation_width"] = 1;
    // do not found any default value for kernels
    onnxOperatorAttribute["kernel_height"] = 1;
    onnxOperatorAttribute["kernel_width"] = 1;
    onnxOperatorAttribute["stride_height"] = 1;
    onnxOperatorAttribute["stride_width"] = 1;
  }
  else if (node.op_type() == "BatchNormalization")
  {
    onnxOperatorAttribute["epsilon"] = 1e-5;
    onnxOperatorAttribute["momentum"] = 0.9;
    onnxOperatorAttribute["training_mode"] = 0;
  }
  else if (node.op_type() == "MaxPool")
  {
    // NOTSET 0, SAME_UPPER 1, SAME_LOWER 2, VALID 3
    onnxOperatorAttribute["auto_pad"] = 0;
    onnxOperatorAttribute["pad_top"] = 0;
    onnxOperatorAttribute["pad_bottom"] = 0;
    onnxOperatorAttribute["pad_right"] = 0;
    onnxOperatorAttribute["pad_left"] = 0;
    onnxOperatorAttribute["dilation_height"] = 1;
    onnxOperatorAttribute["dilation_width"] = 1;
    // do not found any default value for kernels
    onnxOperatorAttribute["kernel_height"] = 1;
    onnxOperatorAttribute["kernel_width"] = 1;
    onnxOperatorAttribute["stride_height"] = 1;
    onnxOperatorAttribute["stride_width"] = 1;
    // same as the convolution layer, just two new attribute
    onnxOperatorAttribute["ceil_mode"] = 0; // floor 0 and ceil 1
    onnxOperatorAttribute["storage_order"] = 0;
  }
  else if (node.op_type() == "GlobalAveragePool")
  {
    // TODO
  }
  else if (node.op_type() == "Reshape")
  {
    // TODO
  }
  else
  {
    // TODO: throw
    cout << "this operator is not been implemented yet" << endl;
  }

  // Iterate through the attributes and set the values in palce of default
  // value.
  for (const onnx::AttributeProto& attr : node.attribute())
  {
    if (attr.name() == "alpha") // Gemm, LeakyRelu
    {
      onnxOperatorAttribute["alpha"] = attr.f();
    }
    else if (attr.name() == "beta") // Gemm
    {
      onnxOperatorAttribute["beta"] = attr.f();
    }
    else if (attr.name() == "transA") // Gemm
    {
      onnxOperatorAttribute["transA"] = attr.i();
    }
    else if (attr.name() == "transB") // Gemm
    {
      onnxOperatorAttribute["transB"] = attr.i();
    }
    else if (attr.name() == "auto_pad") // conv, MaxPool
    {
      // by this we will get to wether we have to use autopad or pads
      onnxOperatorAttribute["auto_pad_or_pads"] = 0;
      if (attr.s() == "NOTSET")
      {
        onnxOperatorAttribute["auto_pad"] = 0;
      }
      else if (attr.s() == "SAME_UPPER")
      {
        onnxOperatorAttribute["auto_pad"] = 1;
      }
      else if (attr.s() == "SAME_LOWER")
      {
        onnxOperatorAttribute["auto_pad"] = 2;
      }
      else if (attr.s() == "VALID")
      {
        onnxOperatorAttribute["auto_pad"] = 3;
      }
    }
    else if (attr.name() == "pads") // conv, MaxPool
    {
      // by this we will get to wether we have to use autopad or pads
      onnxOperatorAttribute["auto_pad_or_pads"] = 1;
      onnxOperatorAttribute["pad_top"] = attr.ints(0);
      onnxOperatorAttribute["pad_bottom"] = attr.ints(1);
      onnxOperatorAttribute["pad_right"] = attr.ints(2);
      onnxOperatorAttribute["pad_left"] = attr.ints(3);
    }
    else if (attr.name() == "strides") // conv, MaxPool
    {
      onnxOperatorAttribute["stride_height"] = attr.ints(0);
      onnxOperatorAttribute["stride_width"] = attr.ints(1);
    }
    else if (attr.name() == "kernel_shape") // conv, MaxPool
    {
      onnxOperatorAttribute["kernel_height"] = attr.ints(0);
      onnxOperatorAttribute["kernel_width"] = attr.ints(1);
    }
    else if (attr.name() == "dilations") // conv, MaxPool
    {
      onnxOperatorAttribute["dilation_height"] = attr.ints(0);
      onnxOperatorAttribute["dilation_width"] = attr.ints(1);
    }
    else if (attr.name() == "group") // conv
    {
      onnxOperatorAttribute["group"] = attr.i();
    }
    else if (attr.name() == "ceil_mode") // MaxPool
    {
      onnxOperatorAttribute["ceil_mode"] = attr.i();
    }
    else if (attr.name() == "storage_order") // MaxPool
    {
      onnxOperatorAttribute["storage_order"] = attr.i();
    }
    else if (attr.name() == "epsilon") // BatchNormalization
    {
      onnxOperatorAttribute["epsilon"] = attr.f();
    }
    else if (attr.name() == "momentum") // BatchNormalization
    {
      onnxOperatorAttribute["momentum"] = attr.f();
    }
    else if (attr.name() == "training_mode") // BatchNormalization
    {
      onnxOperatorAttribute["training_mode"] = attr.i();
    }
  }

  return onnxOperatorAttribute;
}

#endif
