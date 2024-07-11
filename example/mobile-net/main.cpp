// information about the onnx mobilenet v7 model can be found
// https://github.com/onnx/models/tree/main/validated/vision/classification/mobilenet

#include "converter.hpp"

int mul(vector<size_t> v){
    size_t a = 1;
    for(size_t element : v){
        a *= element;
    }
    return a;
}
int main()
{
    // loading the onnx graph
    string onnxFilePath = "mobilenetv2-7.onnx";
    onnx::GraphProto graph = getGraph(onnxFilePath);
    mlpack::FFN<> generatedModel = converter(graph);

    // these are needed specificaly for mobilenet model
    vector<int> topoOrderedNodes = get::TopologicallySortedNodes(graph);
    vector<vector<int>> adjencyMatrix = get::AdjencyMatrix(graph);

    // making the image
    vector<double> data(224 * 224 * 3, 1);
    arma::colvec data_vector(data);
    arma::Mat<double> input(data_vector);
    
    //---------------------------------------
    // forward pass layer by layer
    int i = 0;
    map<int, arma::Mat<double>> bufferOutput;
    for (auto layer : generatedModel.Network())
    {
        int nodeIndex = topoOrderedNodes[i];
        arma::Mat<double> output(mul(layer->OutputDimensions()), 1, arma::fill::ones);
        layer->Forward(input, output);
        if(bufferOutput.find(nodeIndex) != bufferOutput.end()){
            output += bufferOutput[nodeIndex];
        }
        input = output;

        // --------------mobilenet specific operation
        if(adjencyMatrix[nodeIndex].size() == 2){
            bufferOutput[adjencyMatrix[nodeIndex][1]] = output;
        }
        // printing the output dimension
        cout << " output dimensions " << i << " " << output.n_rows << endl;
        i++;
    }
}

