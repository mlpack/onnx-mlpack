#include "GlobalAveragePool.hpp"

inline vector<size_t> AddGlobalAveragePool(mlpack::DAGNetwork<> &dag)
{
    // just to get the dimension of the incoming i will be reseting the ffn
    mlpack::DAGNetwork<> dag_ = dag;
    dag_.Reset();
    vector<size_t> dims = dag_.Network().back()->OutputDimensions();

    size_t kernelWidth = dims[0];
    size_t kernelHeight = dims[1];
    size_t strideWidth = dims[0];
    size_t strideHeight = dims[1];
    bool floor = true;
    // if (onnxOperatorAttribute["ceil_mode"] == 1)
    // {
    //     floor = false;
    // }

    // max pooling part
    // mlpack::MeanPooling *meanPooling = new mlpack::MeanPooling(kernelWidth, kernelHeight, strideWidth, strideHeight, floor);
    // layerParameters.push_back(arma::Mat<double>());
    size_t a = dag.Add<mlpack::MeanPooling>(kernelWidth, kernelHeight, strideWidth, strideHeight, floor);
    cout << "Added mlpack::GlobalAveragePool Layer" << endl;
    return {a};
}
