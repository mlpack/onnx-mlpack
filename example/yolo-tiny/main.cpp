#include "converter.hpp"

int main(){
    // generating the onnx graph
    string onnxFilePath = "tinyyolo-v1.3-o8.onnx";
    onnx::GraphProto graph = getGraph(onnxFilePath);

    // getting the model from the graph
    mlpack::FFN<> generatedModel = converter(graph);
    generatedModel.Reset();

    // Extracting image 
    mlpack::data::ImageInfo imageInfo(416, 416, 1, 1);
    string fileName = "image(416-416)/1.jpg";
    arma::Mat<double> imageMatrix;
    mlpack::data::Load<double>(fileName, imageMatrix, imageInfo, false);
    // cout<<"imageMatrix rows "<<imageMatrix.n_rows<<" cols "<<imageMatrix.n_cols<<endl;

    arma::Mat<double> outputMatrix;
    generatedModel.Network()[0]->Forward(imageMatrix, outputMatrix);
    // outputMatrix.print("outputMatrix");

    // playing with dimensions
    cout<<" input dimensions "<<generatedModel.Network()[0]->InputDimensions()<<endl;
    cout<<" output dimensions "<<generatedModel.Network()[0]->OutputDimensions()<<endl;


    return 0;
}