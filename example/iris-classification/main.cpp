#include "converter.hpp"

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

    return 0;
}