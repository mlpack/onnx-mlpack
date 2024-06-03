#include "converter.hpp"

int main(){
    // generating the onnx graph
    string onnxFilePath = "iris_model.onnx";
    onnx::GraphProto graph = getGraph(onnxFilePath);

    // getting the model from the graph
    mlpack::FFN<> ffn = converter(graph);

    return 0;
}