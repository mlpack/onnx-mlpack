#include "converter.hpp"

int main(){
    // generating the onnx graph
    string onnxFilePath = "tinyyolo-v1.3-o8.onnx";
    onnx::GraphProto graph = getGraph(onnxFilePath);

    // getting the model from the graph
    mlpack::FFN<> generatedModel = converter(graph);

    return 0;
}