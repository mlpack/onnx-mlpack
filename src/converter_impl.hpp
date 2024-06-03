#include "converter.hpp"

onnx::GraphProto getGraph(string filePath)
{
    onnx::ModelProto onnxModel;
    std::ifstream in(filePath, std::ios_base::binary);
    onnxModel.ParseFromIstream(&in);
    in.close();
    return onnxModel.graph();
}

mlpack::FFN<> converter(onnx::GraphProto graph)
{
    mlpack::FFN<> ffn;
    string modelInput = get::ModelInput(graph);
    vector<size_t> inputDimension = get::InputDimension(graph, modelInput);

    // Iterating through nodes in topological order
    string nodeInput = modelInput;
    onnx::NodeProto node = get::CurrentNode(graph, nodeInput);
    while (node.op_type() != "")
    {
        map<string, double> onnxOperatorAttribute = OnnxOperatorAttribute(graph, node);
        AddLayer(ffn, graph, node, onnxOperatorAttribute);

        // move to next node
        nodeInput = node.output(0);
        node = get::CurrentNode(graph, nodeInput);
    }
    return ffn;
}