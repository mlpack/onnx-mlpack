#include "helper.hpp"

string get::ModelInput(onnx::GraphProto graph)
{
    vector<string> inputNames;
    vector<string> initializerNames;
    for (auto input : graph.input())
    {
        inputNames.push_back(input.name());
    }
    for (auto initializer : graph.initializer())
    {
        initializerNames.push_back(initializer.name());
    }
    // input for which no initializer will be the modelInput
    for (const auto &element : inputNames)
    {
        if (std::find(initializerNames.begin(), initializerNames.end(), element) == initializerNames.end())
        {
            return element;
        }
    }
    // **** have to put error condition here
    cout << "error in finding the modelInput" << endl;
    return "";
}

vector<size_t> get::InputDimension(onnx::GraphProto graph, string modelInput)
{
    vector<size_t> dimension;
    for (auto input : graph.input())
    {
        if (input.name() == modelInput)
        {
            int dim_size = input.type().tensor_type().shape().dim().size();
            for (int i = 1; i < dim_size; i++)
            {
                dimension.push_back(input.type().tensor_type().shape().dim(i).dim_value());
            }
        }
    }
    return dimension;
}

onnx::NodeProto get::CurrentNode(onnx::GraphProto graph, string nodeInput)
{
    onnx::NodeProto node;
    
    for (onnx::NodeProto node : graph.node()){
        for(int i=0; i<node.input().size(); i++){
            if( nodeInput == node.input(i)){
                return node;
            }
        }
    }
    return node;
}

onnx::TensorProto get::Initializer(onnx::GraphProto graph, string initializerName){
    for( onnx::TensorProto init: graph.initializer()){
        if(initializerName == init.name()){
            return init;
        }
    }
    throw std::runtime_error("No initializer found with name " + initializerName);
}

