#include "Gemm.hpp"

vector<size_t> AddGemm(mlpack::DAGNetwork<> &dag, onnx::GraphProto graph,
                       onnx::NodeProto node, map<string, double> onnxOperatorAttribute)
{
    // mlpack::LinearNoBias *linearNoBias = new mlpack::LinearNoBias(FindOutputDimension(graph, node));
    // mlpack::Add *add = new mlpack::Add();

    size_t a = dag.Add<mlpack::LinearNoBias>(FindOutputDimension(graph, node));
    size_t b = dag.Add<mlpack::Add>();
    dag.Connect(a, b);

    cout << "Added Gemm Layer" << endl;
    return {a, b};

    // getting the weights in correct dimension
    // arma::mat weights = onnxOperatorAttribute["alpha"] * ExtractWeights(graph, node, onnxOperatorAttribute["transB"]);

    // layerParameters.push_back(weights);
    // cout << "Added mlpack::LinearNoBias Layer" << endl;

    // mlpack::Add *add = new mlpack::Add();
    // // getting the biases in correct dimension
    // arma::mat biases = ExtractBiases(graph, node);
    // // biases.print("biases");
    // layerParameters.push_back(biases);

    // mlpack::MultiLayer* multiLayer = new mlpack::MultiLayer();

    // return dag.Add(add);
    // cout << "Added mlpack::Add Layer" << endl;
}

void TransferWeightToGemm(mlpack::DAGNetwork<> &dag, 
    vector<size_t> &layerIndex, 
    onnx::GraphProto &graph, 
    const onnx::NodeProto &node, 
    std::map<std::string, double> onnxOperatorAttribute)
{
    // gemm layer will be a multilayer with LinearNoBias and Add layer
    arma::mat weights = ExtractWeights(graph, node, onnxOperatorAttribute["transB"]);
    dag.Network()[layerIndex[0]]->Parameters() = weights;
    // layer->Network()[0].Parameters() = weights;

    arma::mat biases = ExtractBiases(graph, node);
    dag.Network()[layerIndex[1]]->Parameters() = biases;
    return;
}

size_t FindOutputDimension(onnx::GraphProto graph, onnx::NodeProto node)
{
    // 3rd input name of Gemm node points to onnx Add operator initializer
    string addInitializerName = node.input(2);
    for (onnx::TensorProto initializer : graph.initializer())
    {
        if (initializer.name() == addInitializerName)
        {
            return initializer.dims(0);
        }
    }
    throw std::runtime_error("No initializer for the third input of Gemm node found");
}

arma::mat ExtractWeights(onnx::GraphProto graph, onnx::NodeProto node, bool transposed)
{
    // finding the initializer in which the weights are stored
    string inputName = node.input(1);
    onnx::TensorProto weightInitializer;
    for (auto initializer : graph.initializer())
    {
        if (initializer.name() == inputName)
        {
            weightInitializer = initializer;
        }
    }

    if (weightInitializer.data_type() == onnx::TensorProto::FLOAT)
    {
        vector<float> tensorData(weightInitializer.raw_data().size() / sizeof(float));
        memcpy(tensorData.data(), weightInitializer.raw_data().data(), weightInitializer.raw_data().size());

        // dimension of matrix stored in initializer
        size_t rows = weightInitializer.dims(0);
        size_t cols = weightInitializer.dims(1);
        arma::fvec armaVector(tensorData); // vector form
        // pytorch works on row major format and armadillo works on column major format
        arma::fmat weights = arma::reshape(armaVector, cols, rows).t(); // this is how the weights will looks in pytorch if we print it
        arma::mat Weights = arma::conv_to<arma::mat>::from(weights);

        return transposed ? Weights : Weights.t();
    }
    throw std::runtime_error("error occured at weight extraction in gemm");
}

arma::mat ExtractBiases(onnx::GraphProto graph, onnx::NodeProto node)
{
    // finding the initializer in which biases are stored
    string input_name = node.input(2);
    onnx::TensorProto biasInitializer;
    for (auto initializer : graph.initializer())
    {
        if (initializer.name() == input_name)
        {
            biasInitializer = initializer;
        }
    }

    if (biasInitializer.data_type() == onnx::TensorProto::FLOAT)
    {
        vector<float> tensorData(biasInitializer.raw_data().size() / sizeof(float));
        memcpy(tensorData.data(), biasInitializer.raw_data().data(), biasInitializer.raw_data().size());

        // dimension of matrix stored in initializer
        size_t elements = biasInitializer.dims(0);
        arma::fvec armaVector(tensorData);
        arma::fmat biases = arma::reshape(armaVector, 1, elements);
        arma::mat Biases = arma::conv_to<arma::mat>::from(biases);

        return Biases;
    }
    throw std::runtime_error("error occured at bias extraction in gemm");
}