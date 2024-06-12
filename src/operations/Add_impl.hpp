#include "Relu.hpp"

void AddAdd(mlpack::FFN<> &ffn, onnx::GraphProto graph,
            onnx::NodeProto node, map<string, double> onnxOperatorAttribute/*, vector<size_t> inputDimension*/)
{
    // // input is in column major {W, H, C, N}
    // vector<int> inDimension(4, 1);
    // int i=0;
    // for(int dim : inputDimension)
    // {
    //     indimension[i] = dim;
    //     i++;
    // }
    // int in_W = inDimension[0];
    // int in_H = inDimension[1];
    // int in_C = inDimension[2];
    // int in_N = inDimension[3];

    // vector<int> initializerDimension(4, 1); //{N, C, H, W}
    // for(int i=0; i < initializer.dims().size(); i++)
    // {
    //     initializerDimension[i] = intializer.dims(i);
    // }
    // int init_W = initializerDimension[3];
    // int init_H = initializerDimension[2];
    // int init_C = initializerDimension[1];
    // int init_N = initializerDimension[0];

    // if(in_W != init_W || in_H != init_H || in_C != init_C || in_N != init_N)
    // {
    //     throw std::runtime_error("input and bias dimension does not match, error in Add_impl");
    // }

    ffn.Add(new mlpack::Identity());
    cout << "Added the Add layer" << endl;
}