#include "attribute.hpp"

std::map<std::string, int> AttributeType = {
    {"UNDEFINED", 0},
    {"FLOAT", 1},
    {"INT", 2},
    {"STRING", 3},
    {"TENSOR", 4},
    {"GRAPH", 5},
    {"SPARSE_TENSOR", 11},
    {"FLOATS", 6},
    {"INTS", 7},
    {"STRINGS", 8},
    {"TENSORS", 9},
    {"GRAPHS", 10},
    {"SPARSE_TENSORS", 12},
    {"TYPE_PROTOS", 14}};

// int extract_attr_value(onnx::AttributeProto attr)
// {
//     vector<int> value;

//     if (attr.type() == AttributeType["INT"])
//     {
//         value = attr.i();
//     }
//     else if (attr.type() == AttributeType["FLOAT"])
//     {
//         value = attr.f();
//     }
//     return value;
// }

// vector<int> extract_attr_values(onnx::AttributeProto attr)
// {
//     vector<int> values;
//     if (attr.type() == AttributeType["INTS"])
//     {
//         for (int i = 0; i < attr.ints().size(); i++)
//         {
//             values.push_back(attr.ints(i));
//         }
//     }
//     else if (attr.type() == AttributeType["FLOATS"])
//     {
//         for (int i = 0; i < attr.floats().size(); i++)
//         {
//             values.push_back(attr.floats(i));
//         }
//     }
//     return values;
// }

//     else if (attr.type() == AttributeType["TENSOR"])
//     {
//         // value = numpy_helper.to_array(attr.t)
//     }
//     else if (attr.type() == AttributeType["STRING"])
//     {
//         // value = attr.s.decode()
//     }
//     else if (attr.type() == AttributeType["GRAPH"])
//     {
//         // value = attr.g
//     }
//     else
//     {
//         cout << "this type is not been implemented yet" << endl;
//     }
//     return value;
// }

map<string, double> OnnxOperatorAttribute(onnx::GraphProto graph, onnx::NodeProto node)
{
    map<string, double> onnxOperatorAttribute;
    // set the default values of operator attributes
    if (node.op_type() == "Gemm")
    {
        onnxOperatorAttribute["alpha"] = 1;
        onnxOperatorAttribute["beta"] = 1;
        onnxOperatorAttribute["transA"] = 0;
        onnxOperatorAttribute["transB"] = 0;
    }
    else if (node.op_type() == "Relu")
    {
    }
    else if (node.op_type() == "LeakyRelu")
    {
        onnxOperatorAttribute["alpha"] = 0.01;
    }
    else if (node.op_type() == "Softmax")
    {
    }
    else if (node.op_type() == "Mul")
    {
    }
    else if (node.op_type() == "Conv")
    { // either there will be auto_pad or pads
        // here we will be encoding the auto_pad attribute
        onnxOperatorAttribute["auto_pad"] = 0; // NOTSET 0, SAME_UPPER 1, SAME_LOWER 2, VALID 3
        onnxOperatorAttribute["pad_top"] = 0;
        onnxOperatorAttribute["pad_bottom"] = 0;
        onnxOperatorAttribute["pad_right"] = 0;
        onnxOperatorAttribute["pad_left"] = 0;
        onnxOperatorAttribute["group"] = 1;
        onnxOperatorAttribute["dilation_height"] = 1;
        onnxOperatorAttribute["dilation_width"] = 1;
        onnxOperatorAttribute["kernel_shape_height"] = 1; // do not found any default value for kernels
        onnxOperatorAttribute["kernel_shape_width"] = 1;
        onnxOperatorAttribute["strides_height"] = 1;
        onnxOperatorAttribute["strides_width"] = 1;
    }
    else if(node.op_type() == "BatchNormalization")
    {
        onnxOperatorAttribute["epsilon"] = 1e-5;
        onnxOperatorAttribute["momentum"] = 0.9;
        onnxOperatorAttribute["training_mode"] = 0;
    }
    else if(node.op_type() == "MaxPool")
    {
        onnxOperatorAttribute["auto_pad"] = 0; // NOTSET 0, SAME_UPPER 1, SAME_LOWER 2, VALID 3
        onnxOperatorAttribute["pad_top"] = 0;
        onnxOperatorAttribute["pad_bottom"] = 0;
        onnxOperatorAttribute["pad_right"] = 0;
        onnxOperatorAttribute["pad_left"] = 0;
        onnxOperatorAttribute["group"] = 1;
        onnxOperatorAttribute["dilation_height"] = 1;
        onnxOperatorAttribute["dilation_width"] = 1;
        onnxOperatorAttribute["kernel_shape_height"] = 1; // do not found any default value for kernels
        onnxOperatorAttribute["kernel_shape_width"] = 1;
        onnxOperatorAttribute["stride_height"] = 1;
        onnxOperatorAttribute["stride_width"] = 1;
        // same as the convolution layer, just two new attribute
        onnxOperatorAttribute["ceil_mode"] = 0;
        onnxOperatorAttribute["storage_order"] = 0;
 
    }
    else
    {
        cout << "this operator is not been implemented yet" << endl;
    }

    // iteratre throught the attribute and set the values //
    for (onnx::AttributeProto attr : node.attribute())
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
        else if (attr.name() == "strides")  // conv, MaxPool
        {
            onnxOperatorAttribute["stride_height"] = attr.ints(0);
            onnxOperatorAttribute["stride_width"] = attr.ints(1);
        }
        else if (attr.name() == "kernel_shape")  // conv, MaxPool
        {
            onnxOperatorAttribute["kernel_height"] = attr.ints(0);
            onnxOperatorAttribute["kernel_width"] = attr.ints(1);
        }
        else if (attr.name() == "dilations")  // conv, MaxPool
        {
            onnxOperatorAttribute["dilation_height"] = attr.ints(0);
            onnxOperatorAttribute["dilation_width"] = attr.ints(1);
        }
        else if (attr.name() == "group")  // conv, MaxPool
        {
            onnxOperatorAttribute["group"] = attr.i();
        }
        else if (attr.name() == "ceil_mode")  // MaxPool
        {
            onnxOperatorAttribute["ceil_mode"] = attr.i();
        }
        else if (attr.name() == "storage_order")  // MaxPool
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
