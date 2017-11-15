#include <iostream>
#include <vector>
#include <cstdio>
#include <fstream>
#include <memory>
#include <random>
#include <arpa/inet.h>

#include "graddesc.h"

using namespace std;

static const uint32_t MAGIC_DATA = 0x00000803;
static const uint32_t MAGIC_LABELS = 0x00000801;

static const uint32_t ROWS = 28;
static const uint32_t COLS = 28;

static const char* TRAIN_DATA = "mnist/train-images-idx3-ubyte";
static const char* TRAIN_LABELS = "mnist/train-labels-idx1-ubyte";
static const char* TEST_DATA = "mnist/t10k-images-idx3-ubyte";
static const char* TEST_LABELS = "mnist/t10k-labels-idx1-ubyte";

struct Image {
  vector<vector<uint8_t> > data;
  uint8_t label;
};

struct mnist_network {
  mnist_network(Network* net)
      : net(net), cost(nullptr) {
  }

  ~mnist_network() {
    delete net;
  }

  void SetImage(const Image& img) {
    for (uint32_t r = 0; r < ROWS; r++) {
      for (uint32_t c = 0; c < COLS; c++) {
        inputs[r][c]->SetValue(img.data[r][c] / 255.0);
      }
    }
    for (uint32_t i = 0; i < 10; i++) {
      labels[i]->SetValue(i == img.label ? 1.0 : 0.0);
    }
  }

  uint32_t GetOutputLabel() {
    uint32_t result = 0;
    double result_value = -1;
    for (uint32_t i = 0; i < 10; i++) {
      double value = outputs[i]->GetValue();
      if (value > result_value) {
        result = i;
        result_value = value;
      }
    }
    return result;
  }

  Network* net;
  vector<vector<InputNode*> > inputs;
  vector<InputNode*> labels;
  vector<Node*> outputs;
  Node* cost;
};

static vector<Image> fail(const char* msg) {
  fputs(msg, stderr);
  return vector<Image>();
}

static default_random_engine rng;

static ParameterNode* InitParameter(Network* net, double stddev) {
  normal_distribution<double> dst(0, stddev);
  return new ParameterNode(net, dst(rng));
}

mnist_network* construct_network() {
  Network* net = new Network;
  mnist_network* mnet = new mnist_network(net);

  vector<Node*> last_layer;
  mnet->inputs.resize(ROWS);
  for (uint32_t r = 0; r < ROWS; r++) {
    for (uint32_t c = 0; c < COLS; c++) {
      InputNode* input = new InputNode(net);
      mnet->inputs[r].push_back(input);
      last_layer.push_back(input);
    }
  }

  for (uint32_t i = 0; i < 10; i++) {
    InputNode* label = new InputNode(net);
    mnet->labels.push_back(label);
  }

  vector<uint32_t> layer_sizes = {30, 10};
  for (uint32_t layer_size : layer_sizes) {
    vector<Node*> layer(layer_size);
    for (uint32_t i = 0; i < layer_size; i++) {
      LinearReducer* lr = new LinearReducer(net, InitParameter(net, 1.0));
      for (Node* n : last_layer) {
        lr->AddTerm(n, InitParameter(net, sqrt(1.0 / layer_size)));
      }
      layer[i] = new SigmoidNode(net, lr);
    }
    last_layer.swap(layer);
  }

  mnet->outputs = last_layer;

  SquaredError* cost = new SquaredError(net);
  for (uint32_t i = 0; i < 10; i++) {
    cost->AddTerm(last_layer[i], mnet->labels[i]);
  }
  mnet->cost = cost;

  mnet->net->TopoSort();
  return mnet;
}

vector<Image> read_images(const char* data_path, const char* label_path) {
  ifstream f_data(data_path);
  if (!f_data.is_open()) {
    return fail("Could not open data file\n");
  }

  ifstream f_labels(label_path);
  if (!f_labels.is_open()) {
    return fail("Could not open labels file\n");
  }

  uint32_t data_magic;
  uint32_t label_magic;
  if (!f_data.read((char*)&data_magic, 4)) {
    return fail("Unexpected end of data file\n");
  }
  if (!f_labels.read((char*)&label_magic, 4)) {
    return fail("Unexpected end of labels file\n");
  }

  data_magic = ntohl(data_magic);
  label_magic = ntohl(label_magic);
  if (data_magic != MAGIC_DATA) {
    return fail("Incorrect data file header\n");
  }
  if (label_magic != MAGIC_LABELS) {
    return fail("Incorrect labels file header\n");
  }

  uint32_t data_len;
  uint32_t label_len;
  if (!f_data.read((char*)&data_len, 4)) {
    return fail("Unexpected end of data file\n");
  }
  if (!f_labels.read((char*)&label_len, 4)) {
    return fail("Unexpected end of labels file\n");
  }

  data_len = ntohl(data_len);
  label_len = ntohl(label_len);
  if (data_len != label_len) {
    return fail("Data and label files have different lengths\n");
  }

  uint32_t rows, cols;
  if (!f_data.read((char*)&rows, 4) ||
      !f_data.read((char*)&cols, 4)) {
    return fail("Unexpected end of data file\n");
  }
  rows = ntohl(rows);
  cols = ntohl(cols);

  if (rows != ROWS || cols != COLS) {
    return fail("Unexpected image sizes\n");
  }

  vector<Image> images;
  for (uint32_t i = 0; i < data_len; i++) {
    Image img;
    if (!f_labels.read((char*)&img.label, 1)) {
      return fail("Unexpected end of labels file\n");
    }
    if (9 < img.label) {
      return fail("Unexpected range on image label\n");
    }

    img.data.resize(rows, vector<uint8_t>(cols));
    for (uint32_t i = 0; i < rows; i++) {
      if (!f_data.read((char*)&img.data[i][0], cols)) {
        return fail("Unexpected end of data file\n");
      }
    }
    images.push_back(img);
  }
  return images;
}

int main() {
  vector<Image> train_set = read_images(TRAIN_DATA, TRAIN_LABELS);
  vector<Image> test_set = read_images(TEST_DATA, TEST_LABELS);

  double learning_rate = 0.1;

  train_set.resize(5000);

  mnist_network* mnet = construct_network();
  for (uint32_t epoch = 0; epoch < 100; epoch++) {
    uint32_t correct = 0;
    double avg_error = 0;
    for (const Image& img : train_set) {
      mnet->SetImage(img);
      mnet->net->ComputeValues();
      mnet->net->ComputePartials(mnet->cost);
      mnet->net->UpdateParameters(learning_rate);
      avg_error += mnet->cost->GetValue() / train_set.size();

      if (mnet->GetOutputLabel() == img.label) {
        ++correct;
      }
    }

    printf("Epoch: %u %.6f %d/%zu %.6f\n", epoch, avg_error, correct,
           train_set.size(), 1.0 * correct / train_set.size());
  }

  uint32_t test_correct = 0;
  for (const Image& img : test_set) {
    mnet->SetImage(img);
    mnet->net->ComputeValues();

    if (mnet->GetOutputLabel() == img.label) {
      ++test_correct;
    }
  }
  printf("Test Result %u/%zu %.6f\n", test_correct, test_set.size(),
         1.0 * test_correct / test_set.size());

  delete mnet;
  return 0;
}
