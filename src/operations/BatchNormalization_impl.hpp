#include "BatchNormalization.hpp"

class BatchNorm_ : public mlpack::BatchNormType<arma::mat>
{
public:
    BatchNorm_(
        const size_t minAxis,
        const size_t maxAxis,
        const double eps,
        const bool average,
        const double momentum,
        onnx::GraphProto &graph,
        const onnx::NodeProto &node) : BatchNormType<arma::mat>(minAxis, maxAxis, eps, average, momentum)
    {
        // setting the trained parameters
        string mean_input = node.input(3);
        string var_input = node.input(4);

        TrainingMean() = arma::conv_to<arma::mat>::from(get::ConvertToColumnMajor(get::Initializer(graph, mean_input)));
        TrainingVariance() = arma::conv_to<arma::mat>::from(get::ConvertToColumnMajor(get::Initializer(graph, var_input)));
    }
};

vector<size_t> AddBatchNormalization(mlpack::DAGNetwork<> &dag, onnx::GraphProto &graph,
                                     const onnx::NodeProto &node, map<string, double> onnxOperatorAttribute)
{
    // experiment below
    double eps = onnxOperatorAttribute["epsilon"];
    double momentum = onnxOperatorAttribute["momentum"];
    // mlpack follow width, height, channel and we want normalization only along channel
    // mlpack::BatchNorm* batchNorm = new mlpack::BatchNorm(2, 2, eps, false, momentum);

    // setting the trained parameters
    // string scale_input = node.input(1); // gamma
    // string B_input = node.input(2);     // beta
    // string mean_input = node.input(3);
    // string var_input = node.input(4);

    // batchNorm->TrainingMean() = arma::conv_to<arma::mat>::from(get::ConvertToColumnMajor(get::Initializer(graph, mean_input)));
    // batchNorm->TrainingVariance() = arma::conv_to<arma::mat>::from(get::ConvertToColumnMajor(get::Initializer(graph, var_input)));

    // batchNorm->Parameters() = arma::join_cols(arma::conv_to<arma::mat>::from(get::ConvertToColumnMajor(get::Initializer(graph, mean_input))), arma::conv_to<arma::mat>::from(get::ConvertToColumnMajor(get::Initializer(graph, var_input))));
    // layerParameters.push_back(arma::join_cols(arma::conv_to<arma::mat>::from(get::ConvertToColumnMajor(get::Initializer(graph, scale_input))), arma::conv_to<arma::mat>::from(get::ConvertToColumnMajor(get::Initializer(graph, B_input)))));
    // experiment above
    size_t a = dag.Add<BatchNorm_>(2, 2, eps, false, momentum, graph, node);

    cout << "Added mlpack::BatchNormalization Layer" << endl;
    return {a};
}

void TransferWeightToBatchNormalization(mlpack::DAGNetwork<> &dag,
                                        vector<size_t> &layerIndex,
                                        onnx::GraphProto &graph,
                                        const onnx::NodeProto &node,
                                        std::map<std::string, double> onnxOperatorAttribute)
{
    // setting the trained parameters
    string scale_input = node.input(1); // gamma
    string B_input = node.input(2);     // beta
    // string mean_input = node.input(3);
    // string var_input = node.input(4);

    // dag.Network()[layerIndex[0]]->TrainingMean() = arma::conv_to<arma::mat>::from(get::ConvertToColumnMajor(get::Initializer(graph, mean_input)));
    // dag.Network()[layerIndex[0]]->TrainingVariance() = arma::conv_to<arma::mat>::from(get::ConvertToColumnMajor(get::Initializer(graph, var_input)));

    dag.Network()[layerIndex[0]]->Parameters() = arma::join_cols(
        arma::conv_to<arma::mat>::from(get::ConvertToColumnMajor(get::Initializer(graph, scale_input))),
        arma::conv_to<arma::mat>::from(get::ConvertToColumnMajor(get::Initializer(graph, B_input))));
}