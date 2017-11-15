#include "graddesc.h"

#include <cmath>
#include <map>

using namespace std;

Node::Node(Network * net) : value(0), dcost(0) {
  net->AddNode(this);
}

Node::~Node() {
}

void Node::PushPartial(Node* n, double d) {
  n->dcost += d;
}

InputNode::InputNode(Network* net, double val) : Node(net) {
  value = val;
}

ParameterNode::ParameterNode(Network* net, double val) : Node(net) {
  value = val;
  net->AddParameter(this);
}

LinearReducer::LinearReducer(Network* net, Node* bias) : Node(net), bias(bias) {
}

void LinearReducer::AddTerm(Node* a, Node* b) {
  terms.push_back(make_pair(a, b));
}

void LinearReducer::Eval() {
  value = bias->GetValue();
  for (const auto& term : terms) {
    value += term.first->GetValue() * term.second->GetValue();
  }
}

void LinearReducer::PushPartial() {
  Node::PushPartial(bias, dcost);
  for (const auto& term : terms) {
    Node::PushPartial(term.first, dcost * term.second->GetValue());
    Node::PushPartial(term.second, dcost * term.first->GetValue());
  }
}

void LinearReducer::EnumDeps(void (*callback)(Node*, void*), void* ctx) {
  callback(bias, ctx);
  for (const auto& term : terms) {
    callback(term.first, ctx);
    callback(term.second, ctx);
  }
}

SigmoidNode::SigmoidNode(Network* net, Node* input) : Node(net), input(input) {
}

void SigmoidNode::Eval() {
  double iv = input->GetValue();
  value = 1.0 / (1.0 + exp(-iv));
}

void SigmoidNode::PushPartial() {
  double iv = input->GetValue();
  double eiv = exp(iv);
  Node::PushPartial(input, dcost * (eiv / pow(1.0 + eiv, 2)));
}

void SigmoidNode::EnumDeps(void (*callback)(Node*, void*), void* ctx) {
  callback(input, ctx);
}

SquaredError::SquaredError(Network* net) : Node(net) {
}

void SquaredError::AddTerm(Node* a, Node* b) {
  terms.push_back(make_pair(a, b));
}

void SquaredError::Eval() {
  value = 0;
  for (const auto& term : terms) {
    value += pow(term.first->GetValue() - term.second->GetValue(), 2);
  }
}

void SquaredError::PushPartial() {
  for (const auto& term : terms) {
    double a = term.first->GetValue();
    double b = term.second->GetValue();
    Node::PushPartial(term.first, dcost * 2 * (a - b));
    Node::PushPartial(term.second, dcost * 2 * (b - a));
  }
}

void SquaredError::EnumDeps(void (*callback)(Node*, void*), void* ctx) {
  for (const auto& term : terms) {
    callback(term.first, ctx);
    callback(term.second, ctx);
  }
}

void Network::AddNode(Node* n) {
  nodes.push_back(unique_ptr<Node>(n));
}

void Network::AddParameter(ParameterNode* n) {
  parameters.push_back(n);
}

void Network::ComputeValues() {
  for (size_t i = 0; i < nodes.size(); i++) {
    nodes[i]->dcost = 0;
    nodes[i]->Eval();
  }
}

void Network::ComputePartials(Node* cost) {
  cost->dcost = 1;
  for (size_t i = nodes.size() - 1; (ssize_t)i >= 0; i--) {
    nodes[i]->PushPartial();
  }
}

void Network::UpdateParameters(double learning_rate) {
  for (auto* param : parameters) {
    param->value -= learning_rate * param->dcost;
  }
}

struct topo_context {
  map<Node*, uint32_t> out_degree;
  vector<Node*> ordered_nodes;
};

void Network::TopoSort() {
  topo_context ctx;
  for (unique_ptr<Node>& n : nodes) {
    n->EnumDeps([](Node* dep, void* pctx) {
      topo_context* ctx = reinterpret_cast<topo_context*>(pctx);
      ++ctx->out_degree[dep];
    }, &ctx);
  }
  for (unique_ptr<Node>& n : nodes) {
    if (ctx.out_degree[n.get()] == 0) {
      ctx.ordered_nodes.push_back(n.get());
    }
  }
  for (uint32_t i = 0; i < ctx.ordered_nodes.size(); i++) {
    ctx.ordered_nodes[i]->EnumDeps([](Node* dep, void* pctx) {
      topo_context* ctx = reinterpret_cast<topo_context*>(pctx);
      if (--ctx->out_degree[dep] == 0) {
        ctx->ordered_nodes.push_back(dep);
      }
    }, &ctx);
  }

  if (ctx.ordered_nodes.size() < nodes.size()) {
    fprintf(stderr, "Topological sort failed: loop detected\n");
  } else if (ctx.out_degree.size() > nodes.size()) {
    fprintf(stderr, "Topological sort failed: out of network connections\n");
  } else {
    for (auto& n : nodes) {
      n.release();
    }
    nodes.clear();

    for (uint32_t i = ctx.ordered_nodes.size() - 1; (int32_t)i >= 0; i--) {
      AddNode(ctx.ordered_nodes[i]);
    }
  }
}
