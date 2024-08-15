// information about the onnx mobilenet v7 model can be found
// https://github.com/onnx/models/tree/main/validated/vision/classification/mobilenet

#include "converter.hpp"
#include <cmath>
#include "class.hpp"

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

// normalize
vector<double> mean_vec = {0.485, 0.456, 0.406};
vector<double> stddev_vec = {0.229, 0.224, 0.225};

int main()
{
    // loading the onnx graph
    string onnxFilePath = "mobilenetv2-7.onnx";
    onnx::GraphProto graph = getGraph(onnxFilePath);
    mlpack::FFN<> generatedModel = converter(graph);

    // these are needed specificaly for mobilenet model
    vector<int> topoOrderedNodes = get::TopologicallySortedNodes(graph);
    vector<vector<int>> adjencyMatrix = get::AdjencyMatrix(graph);

    for (int z = 1; z < 11; z++)
    {
        // // image from the csv file
        string loat_path = "/home/kumarutkarsh/Desktop/onnx-mlpack/example/mobile-net/csv_images/" + to_string(z) + ".csv";
        arma::mat data;
        bool load_status = data.load(loat_path, arma::csv_ascii);
        vector<double> v = convertToColMajor(data, {224, 224, 3});

        arma::mat img(v);
        // img.submat(0, 0, 10, 0).print("input");
        arma::cube finalImage(img.memptr(), 224, 224, 3, false, true);
        // finalImage.print("final output");
        // return 0;
        // normalizing the image
        for (int i = 0; i < 3; i++)
        {
            finalImage.slice(i) = ((finalImage.slice(i) / 255) - mean_vec[i]) / stddev_vec[i];
        }
        arma::mat imageMatrix = arma::vectorise(finalImage);

        //---------------------------------------
        // forward pass layer by layer
        int i = 0;
        map<int, arma::Mat<double>> bufferOutput;
        arma::mat input = imageMatrix;
        for (auto layer : generatedModel.Network())
        {
            int nodeIndex = topoOrderedNodes[i];
            arma::Mat<double> output(mul(layer->OutputDimensions()), 1, arma::fill::ones);
            layer->Forward(input, output);
            if (bufferOutput.find(nodeIndex) != bufferOutput.end())
            {
                output += bufferOutput[nodeIndex];
            }
            input = output;

            //-------------------------------------
            vector<double> v = convertToRowMajor(input, layer->OutputDimensions());
            std::cout << std::fixed << std::setprecision(10);
            // cout << "output Dimension " << i << layer->OutputDimensions() << endl;
            // A.raw_print(std::cout);
            // for (int i = 0; i < 5; i++)
            // {
            //     cout << v[i] << " ";
            // }
            // cout << endl
            //      << endl;
            //-------------------------------------

            // --------------mobilenet specific operation
            if (adjencyMatrix[nodeIndex].size() == 2)
            {
                bufferOutput[adjencyMatrix[nodeIndex][1]] = output;
            }
            // printing the output dimension
            // cout << " output dimensions " << i << " " << output.submat(0, 0, 10, 0) << endl;
            i++;
        }

        vector<double> probs(input.n_elem);
        copy(input.begin(), input.end(), probs.begin());
        probs = softmax(probs);

        auto itr = max_element(probs.begin(), probs.end());
        int bestClassIds = std::distance(probs.begin(), itr);
        cout<<z<<" "<< class_labels[bestClassIds] << endl;
    }
}

// // forward pass one by one

// arma::Mat<double> input = imageMatrix;
// int i = 1;
// for (auto layer : generatedModel.Network())
// {
//     arma::Mat<double> output(mul(layer->OutputDimensions()), 1, arma::fill::ones);
//     layer->Forward(input, output);
//     input = output;

//     // arma::mat A = input.submat(0, 0, 5, 0);
//     vector<double> v = convertToRowMajor(input, layer->OutputDimensions());

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
