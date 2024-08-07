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

    /*
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
    imageVector = convertToRowMajor(imageMatrix, {W, H, C});
    // cout<<"--->"<<imageVector<<endl;
    for(int i=0; i<10; i++){
        cout<<imageVector[i]<<endl;
    }
    */

    string path = "/home/kumarutkarsh/Desktop/onnx-mlpack/example/yolo-tiny/matrix.csv";
    arma::mat data;
    bool load_status = data.load(path, arma::csv_ascii);
    if(load_status){
        cout<<"loaded successfully "<<endl;
        data.submat(0,0,10,0).print("data");
    }else{
        cout<<"failed"<<endl;
    }
    vector<double> v = convertToColMajor(data, {416, 416, 3});
    arma::mat imageMatrix(v);

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



    return 0;
}

// int main(){
//     string imagePath = "image(416-416)/4.jpg";
//     string finalPath = "first.png";
//     DrawRectangle(imagePath, finalPath, 100, 100, 300, 300, {416, 416, 3});
//     return 0;
// }