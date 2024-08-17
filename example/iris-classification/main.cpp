#include "converter.hpp"



int mul(vector<size_t> v){
    size_t a = 1;
    for(size_t element : v){
        a *= element;
    }
    return a;
}

int main(){
    // generating the onnx graph
    string onnxFilePath = "iris_model.onnx";
    onnx::GraphProto graph = getGraph(onnxFilePath);

    // getting the model from the graph
    mlpack::FFN<> generatedModel = converter(graph);
    // generatedModel.Reset();
    cout<<generatedModel.Parameters().n_rows<<"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"<<endl;


    //! Testing the model
    arma::mat data;
    if(data.load("Iris.csv")){
            cout<< "csv data loaded successfully"<< endl;
    }else{
            cout<< "csv data loading unsuccessfull"<<endl;
    }

    //! extracting features
    arma::mat features = data.submat(1, 1, data.n_rows - 1, 4);

    //! normalizing each features
    features.col(0) = (features.col(0) - features.col(0).min() ) / ( features.col(0).max() - features.col(0).min());
    features.col(1) = (features.col(1) - features.col(1).min() ) / ( features.col(1).max() - features.col(1).min());
    features.col(2) = (features.col(2) - features.col(2).min() ) / ( features.col(2).max() - features.col(2).min());
    features.col(3) = (features.col(3) - features.col(3).min() ) / ( features.col(3).max() - features.col(3).min());
    features = features.t();

    //! making prections
    arma::mat Prediction;
    generatedModel.Predict(features, Prediction);
    Prediction = Prediction.t();
    for (size_t i = 0; i < Prediction.n_rows; ++i) {
        // Find the index of the maximum element in the row
        size_t max_index = Prediction.row(i).index_max();

        // Set all elements in the row to 0
        Prediction.row(i).zeros();

        // Set the maximum element to 1
        Prediction(i, max_index) = 1;
    }
    Prediction.print("prediction");




    // vector<double> v = {5, 1, 1, 1};
    // arma::Mat<double> input(v);
    // // forward pass layer by layer
    // int i=1;
    // for (auto layer : generatedModel.Network())
    // {
    //     arma::Mat<double> output(mul(layer->OutputDimensions()), 1, arma::fill::ones);
    //     layer->Forward(input, output);
    //     input = output;

    //     // arma::mat A = input.submat(0, 0, 5, 0);

    //     // printing the output dimension
    //     // Set precision to 10 decimal places
    //     std::cout << std::fixed << std::setprecision(10);

    //     // Use raw_print to have more control over formatting
    //     // A.raw_print(std::cout);
    //     // cout << " output dimensions " << i << " " << output.n_rows << endl;
    //     cout<<"output Dimension "<<i<<layer->OutputDimensions()<<endl;
    //     input.raw_print(std::cout);
    //     // cout<<A<<endl<<endl;
    //     i++;
    // }

    return 0;
}