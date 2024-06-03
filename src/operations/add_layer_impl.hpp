#include "add_layer.hpp"

void AddLayer(mlpack::FFN<> &ffn, onnx::GraphProto graph,
              onnx::NodeProto node, std::map<string, double> onnxoperatorAttribute){
    
    if(node.op_type() == "Gemm"){
        AddGemm(ffn, graph, node, onnxoperatorAttribute);
    }
    else if( node.op_type() == "Softmax"){
        AddSoftmax(ffn);
    }
    else if (node.op_type() == "Relu"){
        AddRelu(ffn);
    }
}