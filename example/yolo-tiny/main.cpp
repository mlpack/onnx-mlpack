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
    std::cout << std::fixed << std::setprecision(10);

    // generating the onnx graph
    string onnxFilePath = "tinyyolo-v2.3-o8.onnx";
    onnx::GraphProto graph = getGraph(onnxFilePath);

    // getting the model from the graph
    mlpack::FFN<> generatedModel = converter(graph);
    cout<<generatedModel.Parameters().n_rows<<"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"<<endl;


    // Extracting image, Input
    mlpack::data::ImageInfo imageInfo(416, 416, 3, 1);
    string fileName = "image(416-416)/10.jpg";
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
    arma::mat imageMatrix(imageVector);

    //---------------------------------------
    
    arma::mat B = imageMatrix.submat(0, 0, 5, 0);
    cout<<"image"<<endl;
    cout<<B<<endl;



    arma::Mat<double> input = imageMatrix;
    // forward pass layer by layer
    int i=1;
    for (auto layer : generatedModel.Network())
    {
        arma::Mat<double> output(mul(layer->OutputDimensions()), 1, arma::fill::ones);
        layer->Forward(input, output);
        input = output;

        arma::mat A = input.submat(0, 0, 5, 0);

        // printing the output dimension
        // Set precision to 10 decimal places
        std::cout << std::fixed << std::setprecision(10);

        // Use raw_print to have more control over formatting
        // A.raw_print(std::cout);
        // cout << " output dimensions " << i << " " << output.n_rows << endl;
        cout<<"output Dimension "<<i<<layer->OutputDimensions()<<endl;
        A.raw_print(std::cout);
        // cout<<A<<endl<<endl;
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
