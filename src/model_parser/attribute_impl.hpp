#include "attribute.hpp"

map<string, double> OnnxOperatorAttribute(onnx::GraphProto &graph, onnx::NodeProto &node)
{
    // Define default values for ONNX node attributes based on operator types.
    // If the node does not specify certain attributes, these default values will be used.
    // link: https://github.com/Talmaj/onnx2pytorch/blob/master/onnx2pytorch/convert/attribute.py
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
    else if (node.op_type() == "Add")
    {
    }
    else if (node.op_type() == "Conv")
    { // either there will be auto_pad or pads, but we will set default value for both of them
        // here we will be encoding the auto_pad attribute
        onnxOperatorAttribute["paddingType"] = 0; // none 0, valid 1, same 2
        onnxOperatorAttribute["auto_pad"] = 0;    // NOTSET 0, SAME_UPPER 1, SAME_LOWER 2, VALID 3
        onnxOperatorAttribute["pad_top"] = 0;
        onnxOperatorAttribute["pad_bottom"] = 0;
        onnxOperatorAttribute["pad_right"] = 0;
        onnxOperatorAttribute["pad_left"] = 0;
        onnxOperatorAttribute["group"] = 1;
        onnxOperatorAttribute["dilation_height"] = 1;
        onnxOperatorAttribute["dilation_width"] = 1;
        onnxOperatorAttribute["kernel_height"] = 1; // do not found any default value for kernels
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
        onnxOperatorAttribute["auto_pad"] = 0; // NOTSET 0, SAME_UPPER 1, SAME_LOWER 2, VALID 3
        onnxOperatorAttribute["pad_top"] = 0;
        onnxOperatorAttribute["pad_bottom"] = 0;
        onnxOperatorAttribute["pad_right"] = 0;
        onnxOperatorAttribute["pad_left"] = 0;
        onnxOperatorAttribute["dilation_height"] = 1;
        onnxOperatorAttribute["dilation_width"] = 1;
        onnxOperatorAttribute["kernel_height"] = 1; // do not found any default value for kernels
        onnxOperatorAttribute["kernel_width"] = 1;
        onnxOperatorAttribute["stride_height"] = 1;
        onnxOperatorAttribute["stride_width"] = 1;
        // same as the convolution layer, just two new attribute
        onnxOperatorAttribute["ceil_mode"] = 0; // floor 0 and ceil 1
        onnxOperatorAttribute["storage_order"] = 0;
    }
    else if (node.op_type() == "GlobalAveragePool")
    {
    }
    else if (node.op_type() == "Reshape")
    {
    }
    else
    {
        cout << "this operator is not been implemented yet" << endl;
    }

    // iteratre throught the attribute and set the values in palce
    // of default value
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
            onnxOperatorAttribute["auto_pad_or_pads"] = 0; // by this we will get to wether we have to use autopad or pads
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
            onnxOperatorAttribute["auto_pad_or_pads"] = 1; // by this we will get to wether we have to use autopad or pads
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

// inspiration for the below commented code comes from onnx2pytorch
// however this is not being used in the current converter
/*
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

int extract_attr_value(onnx::AttributeProto attr)
{
    vector<int> value;

    if (attr.type() == AttributeType["INT"])
    {
        value = attr.i();
    }
    else if (attr.type() == AttributeType["FLOAT"])
    {
        value = attr.f();
    }
    return value;
}

vector<int> extract_attr_values(onnx::AttributeProto attr)
{
    vector<int> values;
    if (attr.type() == AttributeType["INTS"])
    {
        for (int i = 0; i < attr.ints().size(); i++)
        {
            values.push_back(attr.ints(i));
        }
    }
    else if (attr.type() == AttributeType["FLOATS"])
    {
        for (int i = 0; i < attr.floats().size(); i++)
        {
            values.push_back(attr.floats(i));
        }
    }
    return values;
}

    else if (attr.type() == AttributeType["TENSOR"])
    {
        // value = numpy_helper.to_array(attr.t)
    }
    else if (attr.type() == AttributeType["STRING"])
    {
        // value = attr.s.decode()
    }
    else if (attr.type() == AttributeType["GRAPH"])
    {
        // value = attr.g
    }
    else
    {
        cout << "this type is not been implemented yet" << endl;
    }
    return value;
}
*/