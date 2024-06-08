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
    else if (node.op_type() == "LeakyRelu"){
        AddLeakyRelu(ffn);
    }
    else if (node.op_type() == "Mul"){
        AddMul(ffn, graph, node); // scalar multiplication
    }
    else if (node.op_type() == "Add"){
        AddAdd(ffn);
    }
    else if (node.op_type() == "Conv"){
        AddConv(ffn);
    }
    else if (node.op_type() == "BatchNormalization"){
        AddBatchNormalization(ffn);
    }
    else if (node.op_type() == "MaxPool"){
        AddMaxPool(ffn);
    }
    else if (node.op_type() == "BatchNormalization"){
        AddBatchNormalization(ffn);
    }
}