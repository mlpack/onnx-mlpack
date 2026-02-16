#include "add_layer.hpp"

inline vector<size_t> AddLayer(mlpack::DAGNetwork<> &dag, onnx::GraphProto &graph,
                        const onnx::NodeProto &node, std::map<string, double> onnxOperatorAttribute)
{

    if (node.op_type() == "Gemm")
    {
        return AddGemm(dag, graph, node, onnxOperatorAttribute);
    }
    else if (node.op_type() == "Softmax")
    {
        return AddSoftmax(dag);
    }
    else if (node.op_type() == "Relu")
    {
        return AddRelu(dag);
    }
    else if (node.op_type() == "LeakyRelu")
    {
        return AddLeakyRelu(dag, graph, node, onnxOperatorAttribute);
    }
    else if (node.op_type() == "Mul")
    {
        return AddMul(dag, graph, node, onnxOperatorAttribute); // scalar multiplication
    }
    else if (node.op_type() == "Add")
    {
        return AddAdd(dag, graph, node, onnxOperatorAttribute);
    }
    else if (node.op_type() == "Conv")
    {
        return AddConv(dag, graph, node, onnxOperatorAttribute);
    }
    else if (node.op_type() == "BatchNormalization")
    {
        return AddBatchNormalization(dag, graph, node, onnxOperatorAttribute);
    }
    else if (node.op_type() == "MaxPool")
    {
        return AddMaxPool(dag, graph, node, onnxOperatorAttribute);
    }
        else if (node.op_type() == "GlobalAveragePool")
    {
        AddGlobalAveragePool(dag);
    }
        else if (node.op_type() == "Reshape")
    {
        return AddReshape(dag, graph, node, onnxOperatorAttribute);
    }
    else
    {
        cout << "This layer is not been implemented yet:: " << node.op_type() << endl;
    }
    return vector<size_t>();
}

inline void TransferWeights(mlpack::DAGNetwork<> &dag,
                     vector<size_t> &layerIndex,
                     onnx::GraphProto &graph,
                     const onnx::NodeProto &node,
                     std::map<std::string, double> onnxOperatorAttribute)
{
    if (node.op_type() == "Gemm")
    {
        TransferWeightToGemm(dag, layerIndex, graph, node, onnxOperatorAttribute);
    }
    else if (node.op_type() == "Softmax")
    {
        return;
    }
    else if (node.op_type() == "Relu")
    {
        return;
    }
    else if (node.op_type() == "LeakyRelu")
    {
        return;
    }
    else if (node.op_type() == "Mul")
    {
        return;
        // TransferWeightToMul(layer, node); // scalar multiplication
    }
    else if (node.op_type() == "Add")
    {
        return;
        // TransferWeightToAdd(layer, node);
    }
    else if (node.op_type() == "Conv")
    {
        TransferWeightToConv(dag, layerIndex, graph, node, onnxOperatorAttribute);
    }
    else if (node.op_type() == "BatchNormalization")
    {
        TransferWeightToBatchNormalization(dag, layerIndex, graph, node, onnxOperatorAttribute);
    }
    else if (node.op_type() == "MaxPool")
    {
        return;
        // TransferWeightToMaxPool(layer, node);
    }
    else if (node.op_type() == "GlobalAveragePool")
    {
        return;
        // TransferWeightToGlobalAveragePool(dag);
    }
    else if (node.op_type() == "Reshape")
    {
        return;
    }
    else
    {
        cout << "This layer is not been implemented yet:: " << node.op_type() << endl;
    }
}
