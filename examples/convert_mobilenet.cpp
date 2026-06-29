/**
 * @file convert_mobilenet.cpp
 *
 * Example usage of the ONNX/mlpack converter to convert the MobileNet v2.7
 * network to an mlpack::DAGNetwork<> and use it to classify a few test images.
 */
#include <onnx_mlpack.hpp>

int main()
{
  // The ONNX/mlpack converter also has the simpler Convert(filename) function
  // which could be used instead of the code below, but we want to print some
  // useful things about the ONNX graph before converting, so we use the
  // slightly more complex API.

  const std::string filename = "mobilenetv2-7.onnx";
  onnx::GraphProto graph = onnx_mlpack::Load(filename);

  std::cout << "The ONNX graph in '" << filename << "' has "
      << graph.node_size() << " nodes." << std::endl;

  // Remove unnecessary nodes from the graph.
  onnx_mlpack::Simplify(graph);

  std::cout << "After simplification, the graph has "
      << graph.node_size() << " nodes." << std::endl;

  // Now convert the ONNX graph.
  mlpack::DAGNetwork<> result = onnx_mlpack::Convert(graph);

  std::cout << "Converted mlpack graph has " << result.Network().size()
      << " layers." << std::endl;

  // Load three images using mlpack's batch image loading utilities and classify
  // them.
  std::vector<std::string> filenames;
  for (size_t i = 0; i < 3; ++i)
  {
    std::ostringstream oss;
    oss << "imagenet_scaled_" << i << ".jpg";
    filenames.push_back(oss.str());
  }

  arma::mat images;
  mlpack::ImageOptions opts = mlpack::Fatal;
  mlpack::Load(filenames, images, opts);

  // Preprocess the images: group the color channels, and map to [0, 1].
  images = mlpack::GroupChannels(images, opts) / 255.0;

  std::cout << "Loaded " << filenames.size() << " images with "
      << opts.Channels() << " channels and sizes " << opts.Width() << " x "
      << opts.Height() << "." << std::endl;

  arma::mat predictions;
  result.Predict(images, predictions);

  // Mobilenet has 1000 classes.  Load each of the labels as a categorical
  // matrix for conveniently printing.
  arma::mat labels;
  mlpack::TextOptions labelOpts = mlpack::Categorical + mlpack::CSV;
  mlpack::Load("imagenet_labels.txt", labels, labelOpts);

  // Compute the five most likely classes for each image.
  for (size_t i = 0; i < 3; ++i)
  {
    arma::uvec classScoreIndices =
        arma::sort_index(predictions.col(i), "descend");
    std::cout << "Most likely classes for imagenet_scaled_" << i << ".jpg:"
        << std::endl;

    for (size_t j = 0; j < 5; ++j)
    {
      std::cout << " - "
          << labelOpts.DatasetInfo().UnmapString(classScoreIndices[j], 0)
          << ": score " << predictions(classScoreIndices[j], i) << "."
          << std::endl;
    }
  }
}
