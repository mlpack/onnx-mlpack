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
            for (int i = dim_size - 1; i > 0; i--)
            {
                dimension.push_back(input.type().tensor_type().shape().dim(i).dim_value());
            }
        }
    }
    return dimension;
}

vector<int> get::CurrentNode(onnx::GraphProto graph, string nodeInput)
{
    vector<int> nodes;
    for (int i = 0; i < graph.node().size(); i++)
    {
        for (int j = 0; j < graph.node(i).input().size(); j++)
        {
            if (nodeInput == graph.node(i).input(j))
            {
                nodes.push_back(i);
            }
        }
    }
    return nodes;
    // throw std::runtime_error("No node found whose input is  " + nodeInput);

    // for (onnx::NodeProto node : graph.node())
    // {
    //     for (int i = 0; i < node.input().size(); i++)
    //     {
    //         if (nodeInput == node.input(i))
    //         {
    //             return node;
    //         }
    //     }
    // }
    // return node;
}

onnx::TensorProto get::Initializer(onnx::GraphProto graph, string initializerName)
{
    for (onnx::TensorProto init : graph.initializer())
    {
        if (initializerName == init.name())
        {
            return init;
        }
    }
    throw std::runtime_error("No initializer found with name " + initializerName);
}

arma::mat get::ConvertToColumnMajor(onnx::TensorProto initializer)
{
    // onnx initializer stores data in row major format
    // {N, C, H, W}
    vector<int> rowMajorDim(4, 1);
    int j = 3;
    int n_dims = initializer.dims().size();
    for (int i = n_dims - 1; i >= 0; i--)
    {
        rowMajorDim[j] = initializer.dims(i);
        j--;
    }

    int N = rowMajorDim[0]; // l
    int C = rowMajorDim[1]; // k
    int H = rowMajorDim[2]; // j
    int W = rowMajorDim[3]; // i
    vector<double> colMajorData;
    for (int l = 0; l < W; l++)
    {
        for (int k = 0; k < H; k++)
        {
            for (int j = 0; j < C; j++)
            {
                for (int i = 0; i < N; i++)
                {
                    // int colMajorIndex = l * (H * C * N) + k * (C * N) + j * (N) + i;
                    int colMajorIndex = i + j * (N) + k * (C * N) + l * (H * C * N);
                    int rowMajorIndex = i * (C * H * W) + j * (H * W) + k * (W) + l;
                    colMajorData.push_back(initializer.float_data(rowMajorIndex));
                }
            }
        }
    }

    arma::mat matrix(colMajorData);
    return matrix;
}



vector<int> get::TopologicallySortedNodes(onnx::GraphProto &graph)
{
    int totalNodes = graph.node().size();

    // creating the adjencList and inDegree
    vector<vector<int>> adj(totalNodes);
    for (int i = 0; i < totalNodes; i++)
    {
        for (string output : graph.node(i).output())
        {
            vector<int> nodeIndex = get::CurrentNode(graph, output);
            for (int element : nodeIndex)
            {
                adj[i].push_back(element);
            }
        }
    }

    int visitedNode[totalNodes] = {0};
    stack<int> st;

    for(int i=0; i<totalNodes; i++){
        if(!visitedNode[i]){
            dfs(i, visitedNode, adj, st);
        }
    }
    vector<int> topologicalShort;
    while(!st.empty()){
        topologicalShort.push_back(st.top());
        st.pop();
    }
    return topologicalShort;
}

void dfs(int node, int visitedNode[], vector<vector<int>> adj, stack<int> &st){
    visitedNode[node] = 1;
    for(int neighbouringNode : adj[node]){
        if(!visitedNode[neighbouringNode]){
            dfs(neighbouringNode, visitedNode, adj, st);
        }
    }
    st.push(node);
}
