#ifndef GRADDESC_H
#define GRADDESC_H

#include <vector>
#include <utility>
#include <memory>

class Network;

class Node {
  friend class Network;
public:
  Node(Network* net);
  virtual ~Node();

  double GetValue() { return value; }
  double GetPartial() { return dcost; }

protected:
  virtual void Eval() = 0;
  virtual void PushPartial() = 0;
  virtual void EnumDeps(void (*callback)(Node*, void*), void* ctx) = 0;

  double value;
  double dcost;

  static void PushPartial(Node* n, double d);
};

class InputNode : public Node {
public:
  InputNode(Network* net, double val = 0);

  void SetValue(double val) {
    value = val;
  }

protected:
  virtual void Eval() { }
  virtual void PushPartial() { }
  virtual void EnumDeps(void (*callback)(Node*, void*), void* ctx) { }
};

class ParameterNode : public Node {
public:
  ParameterNode(Network* net, double val = 0);

protected:
  virtual void Eval() { }
  virtual void PushPartial() { }
  virtual void EnumDeps(void (*callback)(Node*, void*), void* ctx) { }
};

class LinearReducer : public Node {
public:
  LinearReducer(Network* net, Node* bias);
  
  void AddTerm(Node* a, Node* b);

protected:
  virtual void Eval();
  virtual void PushPartial();
  virtual void EnumDeps(void (*callback)(Node*, void*), void* ctx);

private:
  Node* bias;
  std::vector<std::pair<Node*, Node*> > terms;
};

class SigmoidNode : public Node {
public:
  SigmoidNode(Network* net, Node* input);

protected:
  virtual void Eval();
  virtual void PushPartial();
  virtual void EnumDeps(void (*callback)(Node*, void*), void* ctx);

private:
  Node* input;
};

class SquaredError : public Node {
public:
  SquaredError(Network* net);

  void AddTerm(Node* a, Node* b);

protected:
  virtual void Eval();
  virtual void PushPartial();
  virtual void EnumDeps(void (*callback)(Node*, void*), void* ctx);

private:
  std::vector<std::pair<Node*, Node*> > terms;
};

class L2Regularizer : public Node {
public:
  L2Regularizer(Network* net, Node* base_cost);
  virtual ~L2Regularizer() {};

  void AddParam(Node* a);

protected:
  virtual void Eval();
  virtual void PushPartial();
  virtual void EnumDeps(void (*callback)(Node*, void*), void* ctx);

private:
  Node* base_cost;
};

class Network {
  friend class Node;
  friend class ParameterNode;
public:

  void TopoSort();

  void ComputeValues();
  void ComputePartials(Node* cost);
  void UpdateParameters(double learning_rate);

  uint32_t NumParameters() { return parameters.size(); }
  Node* GetParameter(uint32_t index) { return parameters[index]; }

private:
  void AddNode(Node* n);
  void AddParameter(ParameterNode* n);

  std::vector<ParameterNode*> parameters;
  std::vector<std::unique_ptr<Node> > nodes;
};

#endif
