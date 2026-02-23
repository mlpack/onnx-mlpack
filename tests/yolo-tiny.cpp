/**
 * @file yolo-tiny.cpp
 *
 * Test that the YOLOv3-Tiny network can be loaded from ONNX correctly.
 */
#include <onnx_mlpack.hpp>
#include "catch.hpp"

using namespace std;
using namespace mlpack;
using namespace onnx_mlpack;

TEST_CASE("test_yolo-tiny_onnx_load", "[yolo-tiny]")
{
  // Load the ONNX network.
  const string onnxFilePath = "tinyyolo-v2.3-o8.onnx";
  onnx::GraphProto graph = GetGraph(onnxFilePath);

  // Convert to mlpack DAGNetwork.
  DAGNetwork<> generatedModel = Convert(graph);

  REQUIRE(generatedModel.Network().size() > 0);
}

// Class names for the YOLO model we have trained.
vector<string> classNames = {
    "Aeroplane", "Bicycle", "Bird", "Boat", "Bottle", "Bus", "Car", "Cat",
    "Chair", "Cow", "Dining Table", "Dog", "Horse", "Motorbike", "Person",
    "Potted Plant", "Sheep", "Sofa", "Train", "TV" };

TEST_CASE("test_yolo_accuracy", "[yolo-tiny]")
{
  // Load the network and convert to mlpack format.
  const string onnxFilePath = "tinyyolo-v2.3-o8.onnx";
  onnx::GraphProto graph = GetGraph(onnxFilePath);
  DAGNetwork<> generatedModel = Convert(graph);

  // Now load images to test with.
  vector<string> filenames;
  for (size_t i = 0; i < 4; ++i)
  {
    std::ostringstream oss;
    oss << "yolo_image_" << i << ".jpg";
    filenames.push_back(oss.str());
  }

  arma::mat images;
  ImageOptions opts;
  opts.Fatal() = true;
  REQUIRE(Load(filenames, images, opts) == true);

  REQUIRE(opts.Width() == 416);
  REQUIRE(opts.Height() == 416);
  REQUIRE(opts.Channels() == 3);

  arma::mat groupedImages = GroupChannels(images, opts);
  arma::mat predictions;
  generatedModel.Predict(groupedImages, predictions);

  // Use the raw predictions to compute the most confidently predicted object.
  int numClasses = 20;
  vector<float> anchors = { 1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11,
                            16.62, 10.52};

  vector<float> confidences;
  vector<int> ids;
  vector<vector<int>> bboxes;
  // Convert the output network format into the format used by mlpack.
  for (size_t i = 0; i < predictions.n_cols; ++i)
  {
    arma::cube outputAlias(predictions.colptr(i), 13, 13, 125, true, true);
    arma::mat boxes(4, 13 * 13 * 5);
    arma::rowvec confidences(13 * 13 * 5);
    arma::Row<size_t> boxLabels(13 * 13 * 5);

    // Iterate over all regions and extract all bounding boxes.
    size_t boxIndex = 0;
    for (int cy = 0; cy < 13; cy++)
    {
      for (int cx = 0; cx < 13; cx++)
      {
        for (int c = 0; c < 5; c++)
        {
          // Reconstruct position of bounding box.
          const double tx = outputAlias(cx, cy, 25 * c + 0);
          const double ty = outputAlias(cx, cy, 25 * c + 1);
          const double tw = outputAlias(cx, cy, 25 * c + 2);
          const double th = outputAlias(cx, cy, 25 * c + 3);

          const size_t x = (cx + (1.0 / (1.0 + exp(-tx)))) * 32;
          const size_t y = (cy + (1.0 / (1.0 + exp(-ty)))) * 32;
          const size_t h = exp(tw) * 32 * anchors[2 * c];
          const size_t w = exp(th) * 32 * anchors[2 * c + 1];

          const size_t x1 = x - (w / 2);
          const size_t y1 = y - (h / 2);
          const size_t x2 = x + (w / 2);
          const size_t y2 = y + (h / 2);

          // Ugly: NMS does not properly support any type...
          boxes.col(boxIndex) = { (double) x1, (double) y1,
                                  (double) x2, (double) y2 };

          arma::vec classProbs = vectorise(
              outputAlias.subcube(cx, cy, 25 * c + 5, cx, cy, 25 * c + 24));
          // Apply softmax.
          classProbs = exp(classProbs - classProbs.max());
          classProbs /= accu(classProbs);

          // Compute prediction and confidence.
          boxLabels[boxIndex] = classProbs.index_max();
          confidences[boxIndex] = classProbs[boxLabels[boxIndex]] *
              (1.0 / (1.0 + exp(-outputAlias(cx, cy, 25 * c + 4))));

          ++boxIndex;
        }
      }
    }

    // Filter out any boxes that have confidence less than 0.3.
    arma::uvec keepBoxes = find(confidences >= 0.3);
    boxes = boxes.cols(keepBoxes);
    confidences = confidences.cols(keepBoxes);
    boxLabels = boxLabels.cols(keepBoxes);

    // Use non-maximal suppression to filter bounding boxes.
    arma::uvec filteredBoxIndices;
    NMS<true>::Evaluate(boxes, confidences, filteredBoxIndices, 0.3);

    // With an NMS minimum score of 0.3:
    //   yolo_image_0.jpg: person
    //   yolo_image_1.jpg: person
    //   yolo_image_2.jpg: bird ?? ...take anything
    //   yolo_image_3.jpg: person x2

    switch (i)
    {
      case 0:
        REQUIRE(filteredBoxIndices.n_elem == 1);
        REQUIRE(classNames[boxLabels[filteredBoxIndices[0]]] == "Person");
        break;

      case 1:
        REQUIRE(filteredBoxIndices.n_elem == 1);
        REQUIRE(classNames[boxLabels[filteredBoxIndices[0]]] == "Person");
        break;

      case 2:
        REQUIRE(filteredBoxIndices.n_elem == 1);
        REQUIRE(classNames[boxLabels[filteredBoxIndices[0]]] == "Bird");
        break;

      case 3:
        REQUIRE(filteredBoxIndices.n_elem == 2);
        REQUIRE(classNames[boxLabels[filteredBoxIndices[0]]] == "Person");
        REQUIRE(classNames[boxLabels[filteredBoxIndices[1]]] == "Person");
        break;

      default:
        break;
    }
  }
}
