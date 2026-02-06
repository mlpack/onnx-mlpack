#include "Reshape.hpp"

vector<size_t> AddReshape(mlpack::DAGNetwork<> &dag, onnx::GraphProto graph,
                onnx::NodeProto node, map<string, double> onnxOperatorAttribute)
{
    vector<int> requiredDimensions = FindReshapedDimension(graph, node);
    // vector<size_t> outputDimension(requireDimensions.begin() + 0, requireDimensions.begin() + 3);
    vector<size_t> outputDimension(3, 1);
    // just to get the dimension of the incoming i will be reseting the ffn
    mlpack::DAGNetwork<> dag_ = dag;
    dag_.Reset();
    vector<size_t> inputDimension = dag_.Network().back()->OutputDimensions();

    for (int i = 0; i < 3; i++)
    {
        if (requiredDimensions[i] == 0)
        {
            outputDimension[i] = inputDimension[i];
        }
        else if(requiredDimensions[i] > 0){
            outputDimension[i] = requiredDimensions[i];
        }
    }
    for (int i = 0; i < 3; i++)
    {
        if (requiredDimensions[i] == -1)
        {
            vector<size_t> rough = outputDimension;
            rough[i] = 1;
            outputDimension[i] = std::accumulate(inputDimension.begin(), inputDimension.end(), 1, std::multiplies<int>()) / std::accumulate(rough.begin(), rough.end(), 1, std::multiplies<int>());
        }
    }

    // Reshape *reshape = new Reshape(outputDimension);
    // layerParameters.push_back(arma::Mat<double>());
    size_t a =  dag.Add<Reshape>(outputDimension);
    cout << "Added mlpack::Reshape Layer" << endl;
    return {a};
}

vector<int> FindReshapedDimension(onnx::GraphProto graph, onnx::NodeProto node)
{

    string initializerName = node.input(1); // from this initializer we will be getting the dimension of the output of layer
    onnx::TensorProto initializer = get::Initializer(graph, initializerName);

    vector<int> dimensions(4, 1); /// colMajor W, H, C, C

    int j = 0;
    for (int i = initializer.dims(0) - 1; i >= 0; i--)
    {
        dimensions[j] = initializer.int64_data(i);
        j++;
    }

    return dimensions;
}