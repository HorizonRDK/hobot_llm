// Copyright (c) 2023, Horizon Robotics.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef BPU_BLOOM_MODEL_H_
#define BPU_BLOOM_MODEL_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "easy_dnn/data_structure.h"
#include "easy_dnn/model.h"
#include "easy_dnn/model_manager.h"
#include "easy_dnn/task_manager.h"
#include "tokenizer/tokenization_bloom.h"

using hobot::easy_dnn::DNNTensor;
using hobot::easy_dnn::Model;
using hobot::easy_dnn::ModelManager;
using hobot::easy_dnn::TaskManager;

namespace llm {

class BpuBloomModel {
 public:
  BpuBloomModel() = default;
  ~BpuBloomModel();
  void Read(const std::string& model_dir, const std::string& tokenizer_dir);
  void Reset();
  static void AllocMemory(const std::shared_ptr<Model>& model,
                          std::vector<std::shared_ptr<DNNTensor>>* input,
                          std::vector<std::shared_ptr<DNNTensor>>* output);
  void GetInputOutputInfo(
      const std::shared_ptr<Model>& model,
      const std::vector<std::shared_ptr<DNNTensor>>& input_tensors,
      const std::vector<std::shared_ptr<DNNTensor>>& output_tensors);
  int GetCacheSize() { return cache_size_; }
  int GetVocabSize() { return vocab_size_; }
  void Tokenize(const std::string& query, std::vector<int>& token_ids);
  void ConvertIds2String(const std::vector<int>& token_ids, std::string& text);
  void ConvertId2Token(int token_id, std::string& token);
  void Forward(int cur_token_id, std::vector<float>* next_token_prob);

 private:
  // metadatas
  int chunk_size_ = 1;
  int cache_size_ = 32;
  int hidden_dim_ = 2048;
  int vocab_size_ = 46145;
  int head_ = 16;
  int dk_ = 128;
  int step_ = 1;
  int layers_per_block_ = 6;
  std::vector<std::string> submodel_names_ = {
      "emb", "blocks_0", "blocks_1", "blocks_2", "blocks_3", "head"};
  std::vector<int> cache_index_ = {1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};
  std::vector<float> mask_;
  std::vector<float> alibi_;
  // cache_: 4blocks->12kvcaches->(1, head, dk, cache_size)
  std::vector<std::vector<std::vector<float>>> cache_;

  // models
  std::unordered_map<std::string, std::shared_ptr<Model>> submodels_;

  // input/output tensors
  std::unordered_map<std::string, std::vector<std::shared_ptr<DNNTensor>>>
      submodel_input_tensors_, submodel_output_tensors_;

  // tokenizer
  std::shared_ptr<BloomTokenizer> bloom_tokenizer_;
};

}  // namespace llm

#endif  // BPU_BLOOM_MODEL_H_
