#pragma once

#include "param.h"
#include "result.h"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <list>
#include <new>
#include <set>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace sglang {

namespace ngram {

struct TrieNode {
  std::unordered_map<int32_t, TrieNode*> child;
  std::list<TrieNode*>::const_iterator global_lru_pos;
  std::list<TrieNode*>::const_iterator parent_lru_pos;
  int32_t token;
  TrieNode* parent;
  std::list<TrieNode*> lru;
  int32_t freq = 0;




  uint64_t version = 1;

  struct CompareByFreq {
    bool operator()(TrieNode* a, TrieNode* b) const {
      return std::tie(b->freq, a->token, a) < std::tie(a->freq, b->token, b);
    }
  };
  std::multiset<TrieNode*, CompareByFreq> sorted_children;
};




struct NodeRef {
  TrieNode* ptr = nullptr;
  uint64_t version = 0;
};




struct MatchState {
  uint64_t trie_epoch = 0;
  size_t processed_total_len = 0;
  std::vector<NodeRef> anchors;
};

class Trie {
 public:
  Trie(size_t capacity, const Param& param);

  void insert(const int32_t* tokens, size_t len);

  Result buildRecency(
      const int32_t* context,
      size_t len,
      int32_t last_token,
      size_t draft_token_num,
      const Param& param,
      MatchState& state,
      size_t total_len) const;

  Result buildFrequency(
      const int32_t* context,
      size_t len,
      int32_t last_token,
      size_t draft_token_num,
      const Param& param,
      MatchState& state,
      size_t total_len) const;

  void squeeze(size_t count);

  void reset();

 private:




  std::vector<std::pair<const TrieNode*, int32_t>>
  match(const int32_t* context, size_t len, MatchState& state, size_t total_len) const;



  void rebuildMatchState_(const int32_t* context, size_t len, MatchState& state, size_t total_len) const;


  bool advanceMatchState_(MatchState& state, const int32_t* tokens, size_t len, size_t total_len) const;


  bool validateMatchState_(const MatchState& state) const;



  std::vector<std::pair<const TrieNode*, int32_t>> getExpandableAnchors_(const MatchState& state) const;


  const TrieNode* resolve(const MatchState& state, const NodeRef& ref) const;
  NodeRef rootRef() const {
    return NodeRef{root_, root_->version};
  }
  NodeRef capture(TrieNode* node) const {
    if (node == nullptr) {
      return {};
    }
    return NodeRef{node, node->version};
  }
  void retireNode(TrieNode* node) {
    if (node != nullptr) {
      ++node->version;
    }
  }

  TrieNode* getNode() {
    auto node = node_pool_[--free_node_count_];
    auto version = node->version;
    node->~TrieNode();
    new (node) TrieNode();
    node->version = version;
    return node;
  }

  std::vector<TrieNode> nodes_;
  std::vector<TrieNode*> node_pool_;
  size_t free_node_count_;
  std::list<TrieNode*> global_lru_;
  TrieNode* root_;
  std::vector<TrieNode*> path_;
  Param param_;
  uint64_t trie_epoch_ = 1;
};

}

}
