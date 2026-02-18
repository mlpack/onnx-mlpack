#include "helper.hpp"

inline string get::ModelInput(onnx::GraphProto &graph)
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
        if (std::find(initializerNames.begin(), initializerNames.end(),
                      element) == initializerNames.end())
        {
            return element;
        }
    }
    // **** have to put error condition here
    cout << "error in finding the modelInput" << endl;
    return "";
}

inline vector<size_t> get::InputDimension(onnx::GraphProto &graph,
                                   const string &modelInput)
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

// to get the indexs of all the nodes which will be taking the output of the current node
inline vector<int> get::DependentNodes(onnx::GraphProto &graph,
                                const string &nodeInput)
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
}

inline vector<vector<int>> get::AdjacencyMatrix(onnx::GraphProto &graph)
{
    int totalNodes = graph.node().size();

    // creating the adjencList and inDegree
    vector<vector<int>> adj(totalNodes);
    for (int i = 0; i < totalNodes; i++)
    {
        for (string output : graph.node(i).output())
        {
            vector<int> nodeIndex = get::DependentNodes(graph,
                                                        output);
            for (int element : nodeIndex)
            {
                adj[i].push_back(element);
            }
        }
    }
    return adj;
}

inline vector<int> get::TopologicallySortedNodes(onnx::GraphProto &graph, vector<vector<int>> &adj)
{
    int totalNodes = graph.node().size();

    // creating the adjencList and inDegree
    // vector<vector<int>> adj = get::AdjacencyMatrix(graph);

    // ------ print the adj
    // int i = 0;
    // for (auto element : adj)
    // {
    //     cout << i << " => " << element << endl;
    //     i++;
    // }

    vector<int> visitedNode(totalNodes, 0);

    stack<int> st;

    for (int i = 0; i < totalNodes; i++)
    {
        if (!visitedNode[i])
        {
            get::dfs(i, visitedNode, adj, st);
        }
    }
    vector<int> topologicalShort;
    while (!st.empty())
    {
        topologicalShort.push_back(st.top());
        st.pop();
    }
    return topologicalShort;
}

inline void get::dfs(int node, vector<int>& visited,
              const vector<vector<int>> &adj, stack<int> &st)
{
    visited[node] = 1;
    for (int neighbouringNode : adj[node])
    {
        if (!visited[neighbouringNode])
        {
            dfs(neighbouringNode, visited, adj, st);
        }
    }
    st.push(node);
}

inline const onnx::TensorProto &get::Initializer(onnx::GraphProto &graph,
                                          const string &initializerName)
{
    for (const onnx::TensorProto &init : graph.initializer())
    {
        if (initializerName == init.name())
        {
            return init;
        }
    }
    throw std::runtime_error("No initializer found with name " +
                             initializerName);
}

inline arma::fmat get::ConvertToColumnMajor(const onnx::TensorProto &initializer)
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
    vector<float> colMajorData;

    for (int l = 0; l < N; l++)
    {
        for (int k = 0; k < C; k++)
        {
            for (int j = 0; j < W; j++)
            {
                for (int i = 0; i < H; i++)
                {
                    // int colMajorIndex = l * (H * C * N) + k * (C * N) + j * (N) + i;
                    int rowMajorIndex = j + (i * W) + (k * W * H) + (l * C * W * H);
                    colMajorData.push_back(initializer.float_data(rowMajorIndex));
                }
            }
        }
    }

    arma::fmat matrix(colMajorData);
    return matrix;
}

inline vector<double> get::ConvertToRowMajor(const arma::mat &matrix,
                                      const vector<size_t> &outputDimension)
{
    int C = outputDimension[2];
    int H = outputDimension[1];
    int W = outputDimension[0];

    vector<double> returnValue;
    for (int i = 0; i < C; i++)
    {
        for (int j = 0; j < H; j++)
        {
            for (int k = 0; k < W; k++)
            {
                returnValue.push_back(matrix((j + (H * k) +
                                              (i * W * H)),
                                             0));
            }
        }
    }
    return returnValue;
}
