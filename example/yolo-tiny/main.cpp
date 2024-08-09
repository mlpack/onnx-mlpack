#include "converter.hpp"
#include <cmath>

int mul(vector<size_t> v)
{
    size_t a = 1;
    for (size_t element : v)
    {
        a *= element;
    }
    return a;
}

double sigmoid(double x)
{
    return 1.0 / (1 + exp(-x));
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

int main()
{
    std::cout << std::fixed << std::setprecision(10);

    // generating the onnx graph
    string onnxFilePath = "tinyyolo-v2.3-o8.onnx";
    onnx::GraphProto graph = getGraph(onnxFilePath);

    // getting the model from the graph
    mlpack::FFN<> generatedModel = converter(graph);
    cout << generatedModel.Parameters().n_rows << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << endl;

    /*
    // Extracting image, Input
    mlpack::data::ImageInfo imageInfo(416, 416, 3, 1);
    string fileName = "resized_images/10.png";
    arma::Mat<double> imageMat;
    mlpack::data::Load<double>(fileName, imageMat, imageInfo, false);
    //ImageMatrx => rgb rgb
    // we want int => rrr...ggg...bbb...
    int H = 416;
    int W = 416;
    int C = 3;
    vector<double> imageVector(H*W*C, 0);
    for(int i=0; i<C; i++){
        for(int j=0; j<W; j++){
            for(int k=0; k<H; k++){
                imageVector[k + (j*H) + (i*H*W)] = imageMat(i + (C*W*k) + (C*j), 0);
            }
        }
    }
    arma::mat imageMatrix_(imageVector);
    vector<double> v = convertToRowMajor(imageMatrix_, {W, H, C});
    // cout<<"--->"<<imageVector<<endl;
    for(int i=0; i<10; i++){
        cout<<imageVector[i]<<endl;
    }
    */



   // // image from the csv file
    string path = "/home/kumarutkarsh/Desktop/onnx-mlpack/example/yolo-tiny/matrix.csv";
    arma::mat data;
    bool load_status = data.load(path, arma::csv_ascii);
    vector<double> v = convertToColMajor(data, {416, 416, 3});
    arma::mat imageMatrix(v);


    // forward pass one by one
    /*
    arma::Mat<double> input = imageMatrix;
    int i=1;
    for (auto layer : generatedModel.Network())
    {
        arma::Mat<double> output(mul(layer->OutputDimensions()), 1, arma::fill::ones);
        layer->Forward(input, output);
        input = output;

        // arma::mat A = input.submat(0, 0, 5, 0);
        vector<double> v = convertToRowMajor(input, layer->OutputDimensions());


        // printing the output dimension
        // Set precision to 10 decimal places
        std::cout << std::fixed << std::setprecision(10);

        // Use raw_print to have more control over formatting
        // A.raw_print(std::cout);
        // cout << " output dimensions " << i << " " << output.n_rows << endl;
        cout<<"output Dimension "<<i<<layer->OutputDimensions()<<endl;
        // A.raw_print(std::cout);
        for(int i=0; i<5; i++){
            cout<<v[i]<<" ";
        }
        cout<<endl<<endl;
        // cout<<A<<endl<<endl;
        i++;
    }
    */

    // // // // getting the output from the prediction method
    arma::mat output;
    generatedModel.Predict(imageMatrix, output);
    arma::cube finalOutput(output.memptr(), 13, 13, 125, false, true);

    finalOutput.print("final output");

    // get the most confident object
    int numClasses = 20;
    for (int cy = 0; cy < 13; cy++)
    {
        for (int cx = 0; cx < 13; cx++)
        {
            map<double, vector<double>> confidence_probablity;
            for (int c = 0; c < 5; c++)
            {
                int channel = c * (numClasses + 5);

                double tx = finalOutput(cx, cy, channel + 0);
                double ty = finalOutput(cx, cy, channel + 1);
                double tw = finalOutput(cx, cy, channel + 2);
                double th = finalOutput(cx, cy, channel + 3);
                double tc = finalOutput(cx, cy, channel + 4);

                // int x = (cx + sigmoid(tx)) * 32;
                // int y = (cy + sigmoid(ty)) * 32;
                // int h = 
                // int w = 

                double confidence = sigmoid(tc);
                vector<double> probablity_list;
                for(int i=0; i<20; i++) probablity_list.push_back(finalOutput(cx, cy, channel+5+i));
                probablity_list = softmax(probablity_list);
                confidence_probablity[confidence] = probablity_list;
            }

            double max_confidence = DBL_MIN;
            vector<double> best_probablity(20, 0);

            for(auto element : confidence_probablity){
                if(element.first > max_confidence){
                    max_confidence = element.first;
                    best_probablity = element.second;
                }
            }

            double max_probablity = DBL_MIN;
            int index = 0;
            for(int i=0; i<best_probablity.size(); i++){
                if(best_probablity[i] > max_probablity){
                    index = i;
                    max_probablity = best_probablity[i];
                }
            }

            if(max_confidence > 0.4){
                cout<<"confidence "<<max_confidence<<"("<<cx<<" "<<cy<<")"<<" probablity list "<<index<<endl;
            }
        }
    }

    return 0;
}

// int main(){
//     string imagePath = "image(416-416)/4.jpg";
//     string finalPath = "first.png";
//     DrawRectangle(imagePath, finalPath, 100, 100, 300, 300, {416, 416, 3});
//     return 0;
// }



