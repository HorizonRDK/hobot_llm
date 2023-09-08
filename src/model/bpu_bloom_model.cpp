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

#include "model/bpu_bloom_model.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <utility>

namespace llm {

void BpuBloomModel::GetInputOutputInfo(
    const std::shared_ptr<Model> &model,
    const std::vector<std::shared_ptr<DNNTensor>> &input,
    const std::vector<std::shared_ptr<DNNTensor>> &output) {
  std::string name;
  for (size_t i = 0; i < input.size(); ++i) {
    model->GetInputName(name, i);
    auto &shapes = input[i]->properties.validShape.dimensionSize;
    int32_t &dim = input[i]->properties.validShape.numDimensions;
    std::string layout = (input[i]->properties.tensorLayout ==
                                  hbDNNTensorLayout::HB_DNN_LAYOUT_NHWC
                              ? "NHWC"
                              : "NCHW");
#ifdef SHOW_LOG
    printf("\tInput-%ld: Shape [%d,%d,%d,%d], Layout [%s], %s\n", i, shapes[0],
           shapes[1], shapes[2], shapes[3], layout.c_str(), name.c_str());
#endif
  }
  // Output info
  for (size_t i = 0; i < output.size(); ++i) {
    model->GetOutputName(name, i);
    auto &shapes = output[i]->properties.validShape.dimensionSize;
    int32_t &dim = input[i]->properties.validShape.numDimensions;
    std::string layout = (output[i]->properties.tensorLayout ==
                                  hbDNNTensorLayout::HB_DNN_LAYOUT_NHWC
                              ? "NHWC"
                              : "NCHW");
#ifdef SHOW_LOG
    printf("\tOutput-%ld: Shape [%d,%d,%d,%d], Layout [%s], %s\n", i, shapes[0],
           shapes[1], shapes[2], shapes[3], layout.c_str(), name.c_str());
#endif
  }
}

void BpuBloomModel::Read(const std::string &model_dir,
                         const std::string &tokenizer_dir) {
  for (auto &name : submodel_names_) {
    std::string submodel_path = model_dir + "/" + name + ".bin";
    // 0. Init managers
    ModelManager *model_manager = ModelManager::GetInstance();

    // 1. Load models
    std::vector<Model *> models;
    int ret_code = model_manager->Load(models, submodel_path);
    if (ret_code != 0) {
      throw std::runtime_error("Failed to load file: " + submodel_path);
    }
    std::vector<std::shared_ptr<DNNTensor>> input, output;
    std::shared_ptr<Model> submodel(
        model_manager->GetModel([&name](Model *model) {
          return model->GetName().find(name) != std::string::npos;
        }));
    submodels_[name] = std::move(submodel);
    submodel_input_tensors_[name] = std::move(input);
    submodel_output_tensors_[name] = std::move(output);

    // 2. Init input/output tensors
    AllocMemory(submodels_[name], &(submodel_input_tensors_[name]),
                &(submodel_output_tensors_[name]));

    // 3. Read model input/output nodes
#ifdef SHOW_LOG
    printf("BPU SubModel: %s\n", name.c_str());
#endif
    GetInputOutputInfo(submodels_[name], submodel_input_tensors_[name],
                       submodel_output_tensors_[name]);
  }

  // 4. Parse metadatas
  int chunk_size = (submodel_input_tensors_["blocks_0"][0]
                        ->properties.validShape.dimensionSize[3]);
  int hidden_dim = (submodel_input_tensors_["blocks_0"][0]
                        ->properties.validShape.dimensionSize[1]);
  int head = (submodel_input_tensors_["blocks_0"][1]
                  ->properties.validShape.dimensionSize[1]);
  int dk = (submodel_input_tensors_["blocks_0"][1]
                ->properties.validShape.dimensionSize[2]);
  cache_size_ = (submodel_input_tensors_["blocks_0"][1]
                     ->properties.validShape.dimensionSize[3]);
  int vocab_size = (submodel_output_tensors_["head"][0]
                        ->properties.validShape.dimensionSize[2]);
  assert(chunk_size == chunk_size_);
  assert(hidden_dim == hidden_dim_);
  assert(head == head_);
  assert(dk == dk_);
  assert(vocab_size == vocab_size_);
  assert(submodels_["blocks_0"]->GetInputCount() == 3 + layers_per_block_ * 2);

  // 5. Read alibi tensor
  alibi_.clear();
  alibi_.resize(head_ * (cache_size_ + chunk_size_));
  std::ifstream file(model_dir + "/alibi.bin", std::ios::binary);
  file.read(reinterpret_cast<char *>(alibi_.data()),
            alibi_.size() * sizeof(float));
  assert(file.is_open() && !file.fail());

#ifdef SHOW_LOG
  printf("Bpu Model Info:\n");
  printf("\tchunk_size %d\n", chunk_size_);
  printf("\thidden_dim %d\n", hidden_dim_);
  printf("\thead %d\n", head_);
  printf("\tdk %d\n", dk_);
  printf("\tcache_size %d\n", cache_size_);
  printf("\tvocab_size %d\n", vocab_size_);
  printf("\tlayers_per_block %d\n", layers_per_block_);
#endif
  Reset();

  // 6. Init tokenizer
  std::string py_file_path = tokenizer_dir;
  std::string vocab_file_path = tokenizer_dir + "/bloom_1b4_zh";
  bloom_tokenizer_ = std::make_shared<BloomTokenizer>();
  bool ret = bloom_tokenizer_->Init(py_file_path, vocab_file_path);
  assert(true == ret);
}

void BpuBloomModel::AllocMemory(
    const std::shared_ptr<Model> &model,
    std::vector<std::shared_ptr<DNNTensor>> *inputs,
    std::vector<std::shared_ptr<DNNTensor>> *outputs) {
  int input_counts = model->GetInputCount();
  inputs->resize(input_counts);
  for (size_t i = 0; i < input_counts; ++i) {
    inputs->at(i).reset(new DNNTensor);
    auto &item = inputs->at(i);
    model->GetInputTensorProperties(item->properties, i);
    hbSysAllocCachedMem(&(item->sysMem[0]), item->properties.alignedByteSize);
  }
  int output_counts = model->GetOutputCount();
  outputs->resize(output_counts);
  for (size_t i = 0; i < output_counts; ++i) {
    outputs->at(i).reset(new DNNTensor);
    auto &item = outputs->at(i);
    model->GetOutputTensorProperties(item->properties, i);
    hbSysAllocCachedMem(&(item->sysMem[0]), item->properties.alignedByteSize);
  }
}

void BpuBloomModel::Reset() {
  // Reset input/output tensors with zero
  for (auto &name : submodel_names_) {
    for (auto &tensor : submodel_input_tensors_[name]) {
      memset(tensor->sysMem[0].virAddr, 0, tensor->properties.alignedByteSize);
    }
    for (auto &tensor : submodel_output_tensors_[name]) {
      memset(tensor->sysMem[0].virAddr, 0, tensor->properties.alignedByteSize);
    }
  }
  // mask_: (1, 1, chunk_size, cache_size + chunk_size), reset to -1000.0
  mask_.clear();
  mask_.resize(chunk_size_ * (cache_size_ + chunk_size_), -1000.0);
  // cache_: 4blocks->12kvcaches->(1, head, dk, cache_size), reset to 0.0
  cache_.clear();
  cache_.resize(4, std::vector<std::vector<float>>(
                       (layers_per_block_ * 2),
                       std::vector<float>(head_ * dk_ * cache_size_, 0.0)));
  // alibi_: (1, head, 1, cache_size + chunk_size), do nothing here
  step_ = 1;
}

void BpuBloomModel::Tokenize(const std::string &query,
                             std::vector<int> &token_ids) {
  bloom_tokenizer_->Encode(query, token_ids);
}

void BpuBloomModel::ConvertIds2String(const std::vector<int> &token_ids,
                                      std::string &text) {
  bloom_tokenizer_->Decode(token_ids, text);
}

void BpuBloomModel::ConvertId2Token(int token_id, std::string &token) {
  bloom_tokenizer_->Decode(token_id, token);
}

void BpuBloomModel::Forward(int cur_token_id,
                            std::vector<float> *next_token_prob) {
  TaskManager *task_manager = TaskManager::GetInstance();

  // 1. Forward
  for (int i = 0; i < submodel_names_.size(); ++i) {
    auto &name = submodel_names_[i];
    if (i == 0) {
      // prepare input_id for "emb"
      auto &input_id = submodel_input_tensors_[name][0];
      auto feat_ptr = reinterpret_cast<float *>(input_id->sysMem[0].virAddr);
      *feat_ptr = static_cast<float>(cur_token_id);
    } else {
      // prepare hidden_states for "blocks_x" & "head"
      memcpy(submodel_input_tensors_[name][0]->sysMem[0].virAddr,
             submodel_output_tensors_[submodel_names_[i - 1]][0]
                 ->sysMem[0]
                 .virAddr,
             submodel_output_tensors_[submodel_names_[i - 1]][0]
                 ->properties.alignedByteSize);
      // prepare mask/alibi/cache for "blocks_x"
      if (i != submodel_names_.size() - 1) {
        // 1. input_mask is a float tensor of shape
        //    [1, 1, chunk_size_, cache_size_ + chunk_size_]
        //    mask_ has the same shape with input_mask, we want to do
        //    mask_[-step_] = 0.0 and copy mask_ to input_mask
        auto &input_mask = submodel_input_tensors_[name][3];
        auto input_mask_ptr =
            reinterpret_cast<float *>(input_mask->sysMem[0].virAddr);
        mask_[mask_.size() - step_] = 0.0;
        memcpy(input_mask_ptr, mask_.data(),
               sizeof(float) * (cache_size_ + chunk_size_));
        // 2. input_alibi is a float tensor of shape
        //    [1, head_, 1, cache_size_ + chunk_size_]
        //    alibi_ has the same shape with input_alibi, we want to do
        //    input_alibi[:, :, :, -(step_):] = alibi_[:, :, :, :step_]
        auto &input_alibi = submodel_input_tensors_[name][2];
        auto input_alibi_ptr =
            reinterpret_cast<float *>(input_alibi->sysMem[0].virAddr);
        for (int j = 0; j < head_; ++j) {
          memcpy(input_alibi_ptr + j * (cache_size_ + chunk_size_) +
                     (cache_size_ + chunk_size_ - step_),
                 alibi_.data() + j * (cache_size_ + chunk_size_),
                 sizeof(float) * step_);
        }
        // 3. input_cache is a float tensor of shape
        //    [1, head_, dk_, cache_size_]
        //    cache_[i - 1][k] has the same shape with input_cache, we want
        //    input_cache[:, :, :, -(step_ - 1):] = cache[:, :, :, :(step_-1)]
        for (int k = 0; k < (layers_per_block_ * 2); ++k) {
          if (step_ == 1) break;
          auto &input_cache = submodel_input_tensors_[name][cache_index_[k]];
          auto input_cache_ptr =
              reinterpret_cast<float *>(input_cache->sysMem[0].virAddr);
          // cache_: 4blocks->12kvcaches->(1, head, dk, cache_size)
          // cache_[i - 1][k]: (i - 1)th blocks and its' k-th layercache
          auto &cache = cache_[i - 1][k];
          for (int m = 0; m < head_; ++m) {
            for (int n = 0; n < dk_; ++n) {
              memcpy(input_cache_ptr + m * dk_ * cache_size_ + n * cache_size_ +
                         (cache_size_ - step_ + 1),
                     cache.data() + m * dk_ * cache_size_ + n * cache_size_,
                     sizeof(float) * (step_ - 1));
            }
          }
        }
      }
    }
    for (auto &tensor : submodel_input_tensors_[name]) {
      hbSysFlushMem(&(tensor->sysMem[0]), HB_SYS_MEM_CACHE_CLEAN);
    }
    auto infer_task = task_manager->GetModelInferTask(5000);
    infer_task->SetModel(submodels_[name].get());
    infer_task->SetInputTensors(submodel_input_tensors_[name]);
    infer_task->SetOutputTensors(submodel_output_tensors_[name]);
    infer_task->RunInfer();
    infer_task->WaitInferDone(5000);
    infer_task.reset();
    for (auto &tensor : submodel_output_tensors_[name]) {
      hbSysFlushMem(&(tensor->sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
    }
    if (i != 0 && i != submodel_names_.size() - 1) {
      // update cache
      //    output_cache is a float tensor of shape [1, head_, dk_, 1]
      //    cache_[i - 1][k] has a shape of     [1, head_, dk_, cache_size_]
      //    we want to do: cache[:, :, :, (step_ - 1):(step_)] = output_cache
      for (int k = 0; k < (layers_per_block_ * 2); ++k) {
        auto &output_cache = submodel_output_tensors_[name][k + 1];
        auto output_cache_ptr =
            reinterpret_cast<float *>(output_cache->sysMem[0].virAddr);
        // cache_: 4blocks->12kvcaches->(1, head, dk, cache_size)
        // cache_[i - 1][k]: (i - 1)th blocks and its' (k)th layercache
        auto &cache = cache_[i - 1][k];
        for (int m = 0; m < head_; ++m) {
          for (int n = 0; n < dk_; ++n) {
            cache[m * dk_ * cache_size_ + n * cache_size_ + (step_ - 1)] =
                output_cache_ptr[m * dk_ + n];
          }
        }
      }
    }
  }

  // 2. Extract final outout_prob, [1, 1, vocab_size, 1]
  const float *raw_data = reinterpret_cast<float *>(
      submodel_output_tensors_["head"][0]->sysMem[0].virAddr);
  next_token_prob->clear();
  next_token_prob->resize(vocab_size_);
  memcpy(next_token_prob->data(), raw_data, vocab_size_ * sizeof(float));
  step_++;
}

BpuBloomModel::~BpuBloomModel() {
  for (auto &name : submodel_names_) {
    for (auto &tensor : submodel_input_tensors_[name]) {
      hbSysFreeMem(tensor->sysMem);
    }
    for (auto &tensor : submodel_output_tensors_[name]) {
      hbSysFreeMem(tensor->sysMem);
    }
  }
}

}  // namespace llm
