#include "converter.hpp"

int main(){
    // generating the onnx graph
    string onnxFilePath = "tinyyolo-v1.3-o8.onnx";
    onnx::GraphProto graph = getGraph(onnxFilePath);

    // getting the model from the graph
    mlpack::FFN<> generatedModel = converter(graph);
    // Extracting image 
    mlpack::data::ImageInfo imageInfo(416, 416, 3, 1);
    string fileName = "image(416-416)/1.jpg";
    arma::Mat<double> imageMatrix;
    mlpack::data::Load<double>(fileName, imageMatrix, imageInfo, false);
    // cout<<"imageMatrix rows "<<imageMatrix.n_rows<<" cols "<<imageMatrix.n_cols<<endl;

    arma::Mat<double> outputMatrix;
    generatedModel.Predict(imageMatrix, outputMatrix);
    generatedModel.Network()[2]->Forward(imageMatrix, outputMatrix);
    // outputMatrix.print("outputMatrix");


    int i = 0;
    arma::Mat<double> input = imageMatrix;
    arma::Mat<double> output;
    for(auto layer : generatedModel.Network())
    {
        // layer->Forward(input, output);
        // input = output;
        //printing the output dimension
        cout<<" output dimensions "<<i<<" "<<layer->OutputDimensions()<<endl;
        i++;
    }


    return 0;
}




// arma::Mat<double> input = imageMatrix;
// arma::Mat<double> output;
// for(auto layer : generatedModel.Network())
// {
//     layer->Forward(input, output);
//     input = output;
// }

