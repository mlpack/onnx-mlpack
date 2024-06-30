// information about the onnx mobilenet v7 model can be found
// https://github.com/onnx/models/tree/main/validated/vision/classification/mobilenet

#include "converter.hpp"

int main()
{
    // generating the onnx graph
    string onnxFilePath = "mobilenetv2-7.onnx";
    onnx::GraphProto graph = getGraph(onnxFilePath);
    mlpack::FFN<> generatedModel = converter(graph);

    cout << get::TopologicallySortedNodes(graph) << endl;
    int i = 0;
    for (auto layer : generatedModel.Network())
    {
        // layer->Forward(input, output);
        // input = output;
        // printing the output dimension
        cout << " output dimensions " << i << " " <<graph.node(i).name()<< layer->OutputDimensions() << endl;
        i++;
    }
}
