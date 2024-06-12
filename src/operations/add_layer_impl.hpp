#include "add_layer.hpp"

void AddLayer(mlpack::FFN<> &ffn, onnx::GraphProto graph,
              onnx::NodeProto node, std::map<string, double> onnxoperatorAttribute)
{

    if (node.op_type() == "Gemm")
    {
        AddGemm(ffn, graph, node, onnxoperatorAttribute);
    }
    else if (node.op_type() == "Softmax")
    {
        AddSoftmax(ffn);
    }
    else if (node.op_type() == "Relu")
    {
        AddRelu(ffn);
    }
    else if (node.op_type() == "LeakyRelu")
    {
        AddLeakyRelu(ffn, graph, node, onnxoperatorAttribute);
    }
    else if (node.op_type() == "Mul")
    {
        AddMul(ffn, graph, node, onnxoperatorAttribute); // scalar multiplication
    }
    else if (node.op_type() == "Add")
    {
        AddAdd(ffn, graph, node, onnxoperatorAttribute);
    }
    else if (node.op_type() == "Conv")
    {
        AddConv(ffn, graph, node, onnxoperatorAttribute);
    }
    else if (node.op_type() == "BatchNormalization")
    {
        AddBatchNormalization(ffn, graph, node, onnxoperatorAttribute);
    }
    else if (node.op_type() == "MaxPool")
    {
        AddMaxPool(ffn, graph, node, onnxoperatorAttribute);
    }
    else
    {
        cout << "This layer is not been implemented yet" << endl;
    }
}
