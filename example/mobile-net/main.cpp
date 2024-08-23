// information about the onnx mobilenet v7 model can be found
// https://github.com/onnx/models/tree/main/validated/vision/classification/mobilenet

#include "converter.hpp"
#include <cmath>
#include "class.hpp"

// To get the product of element of vector
int mul(vector<size_t> v)
{
    size_t a = 1;
    for (size_t element : v)
    {
        a *= element;
    }
    return a;
}

vector<double> softmax(vector<double> v)
{
    auto max_it = max_element(v.begin(), v.end());
    double max = *max_it;
    double sum = 0;
    for (int i = 0; i < v.size(); i++)
    {
        v[i] = v[i] - max;
        v[i] = exp(v[i]);
        sum += v[i];
    }

    for (int i = 0; i < v.size(); i++)
    {
        v[i] = v[i] / sum;
    }

    return v;
}

// normalizing the image
vector<double> mean_vec = {0.485, 0.456, 0.406};
vector<double> stddev_vec = {0.229, 0.224, 0.225};

int main()
{
    // loading the onnx graph
    string onnxFilePath = "mobilenetv2-7.onnx";
    onnx::GraphProto graph = getGraph(onnxFilePath);
    mlpack::FFN<> generatedModel = converter(graph);

    // these are needed specificaly for mobilenet model for skip connection
    vector<int> topoOrderedNodes = get::TopologicallySortedNodes(graph);
    vector<vector<int>> adjencyMatrix = get::AdjacencyMatrix(graph);

    for (int z = 1; z < 11; z++)
    {
        // // image from the csv file
        string loat_path = "/home/kumarutkarsh/Desktop/onnx-mlpack/example/mobile-net/csv_images/" + to_string(z) + ".csv";
        arma::mat img;
        bool load_status = img.load(loat_path, arma::csv_ascii);
        get::ImageToColumnMajor(img, {224, 224, 3});

        // normalizing the image
        arma::cube finalImage(img.memptr(), 224, 224, 3, false, true);
        for (int i = 0; i < 3; i++)
        {
            finalImage.slice(i) = ((finalImage.slice(i) / 255) -
                                   mean_vec[i]) /
                                  stddev_vec[i];
        }

        // **** mobileNet spedific
        // directly making the prediction will lead to wrong result bcoz
        // while generating the ffn in mlpack, the converter has not considered
        // the skip connection

        // forward pass layer by layer
        int i = 0;
        // storing the output in case residual connection
        map<int, arma::Mat<double>> bufferOutput;
        arma::mat input = arma::vectorise(finalImage);
        for (auto layer : generatedModel.Network())
        {
            int nodeIndex = topoOrderedNodes[i];
            // To store the output of the layer, we first have to specify the
            // size of the output to the matrix that is going to store the output
            arma::Mat<double> output(mul(layer->OutputDimensions()), 1,
                                     arma::fill::ones);
            layer->Forward(input, output);

            // if the layer has incoming residual connection then add the output
            // with the residual output
            if (bufferOutput.find(nodeIndex) != bufferOutput.end())
            {
                output += bufferOutput[nodeIndex];
            }
            input = output;

            // if the layer has out going residual connection then store the output
            // of the layer in buffer
            if (adjencyMatrix[nodeIndex].size() == 2)
            {
                bufferOutput[adjencyMatrix[nodeIndex][1]] = output;
            }
            i++;
        }
        arma::mat &finalOutput = input;
        // converting the output of the model to probablity values
        vector<double> probs(finalOutput.n_elem);
        copy(finalOutput.begin(), finalOutput.end(), probs.begin());
        probs = softmax(probs);

        // considering the highest confidence class to be the output of the model
        auto itr = max_element(probs.begin(), probs.end());
        int bestClassIds = std::distance(probs.begin(), itr);
        cout << z << ".png" << " " << class_labels[bestClassIds] << endl;
    }
}

// // *** below code is for debugging purpose
// // forward pass one by one
// arma::Mat<double> input = imageMatrix;
// int i = 1;
// for (auto layer : generatedModel.Network())
// {
//     arma::Mat<double> output(mul(layer->OutputDimensions()), 1, arma::fill::ones);
//     layer->Forward(input, output);
//     input = output;

//     // arma::mat A = input.submat(0, 0, 5, 0);
//     vector<double> v = ConvertToRowMajor(input, layer->OutputDimensions());

//     // printing the output dimension
//     // Set precision to 10 decimal places
//     std::cout << std::fixed << std::setprecision(10);

//     // Use raw_print to have more control over formatting
//     // A.raw_print(std::cout);
//     // cout << " output dimensions " << i << " " << output.n_rows << endl;
//     cout << "output Dimension " << i << layer->OutputDimensions() << endl;
//     // A.raw_print(std::cout);
//     for (int i = 0; i < 5; i++)
//     {
//         cout << v[i] << " ";
//     }
//     cout << endl
//          << endl;
//     // cout<<A<<endl<<endl;
//     i++;
// }
