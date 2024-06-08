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

int extract_attr_values(onnx::AttributeProto attr)
{
    int value;

    if (attr.type() == AttributeType["INT"])
    {
        value = attr.i();
    }
    else if (attr.type() == AttributeType["FLOAT"])
    {
        value = attr.f();
    }
    else if (attr.type() == AttributeType["INTS"])
    {
        // value = tuple(attr.ints)
    }
    else if (attr.type() == AttributeType["FLOATS"])
    {
        // value = tuple(attr.floats)
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
    else
    {
        cout << "this operator is not been implemented yet" << endl;
    }

    // iteratre throught the attribute and set the values
    for (onnx::AttributeProto attr : node.attribute())
    {
        if (attr.name() == "alpha") // Gemm, LeakyRelu
        {
            onnxOperatorAttribute["alpha"] = extract_attr_values(attr);
        }
        else if (attr.name() == "beta")
        {
            onnxOperatorAttribute["beta"] = extract_attr_values(attr);
        }
        else if (attr.name() == "transA")
        {
            onnxOperatorAttribute["transA"] = extract_attr_values(attr);
        }
        else if (attr.name() == "transB")
        {
            onnxOperatorAttribute["transB"] = extract_attr_values(attr);
        }
    }

    return onnxOperatorAttribute;
}
