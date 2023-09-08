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

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <string>

#include "model/bpu_bloom_model.h"

#define LLM_COLOR(a, b) "\033[" #b "m" << a << "\033[0m"
#define LLM_GREEN(a) LLM_COLOR(a, 32)
#define LLM_RED(a) LLM_COLOR(a, 31)
#define LLM_PINK(a) LLM_COLOR(a, 35)
#define LLM_YELLOW(a) LLM_COLOR(a, 33)
#define LLM_BLUE(a) LLM_COLOR(a, 34)

int main(int argc, char* argv[]) {
  if (argc < 3) {
    printf("Usage: %s <model_path> <tokenizer_path>\n", argv[0]);
    return -1;
  }
  llm::BpuBloomModel model;
  model.Read(argv[1], argv[2]);

  int cache_size = model.GetCacheSize();
  int vocab_size = model.GetVocabSize();
  std::cout << LLM_RED(
                   "这是一个基于X3派1.5B大模型的"
                   "交互演示Demo，请输入你的问题并按下回车，如需重新开始，请输"
                   "入reset，如需退出"
                   "请输入exit")
            << std::endl;
  std::vector<int> output_ids;
  int prev_token_id = -1;
  while (true) {
    if (output_ids.size() + 20 >= cache_size) {
      model.Reset();
      output_ids.clear();
    }
    std::string query, token;
    std::cout << LLM_BLUE(">>> 用户：");
    fflush(stdout);
    std::getline(std::cin, query);
    fflush(stdin);
    std::cout << LLM_GREEN(">>> 机器人：");
    fflush(stdout);
    if (query == "exit") {
      std::cout << LLM_GREEN("好的，祝您生活愉快，再见~\n");
      fflush(stdout);
      break;
    } else if (query == "reset") {
      model.Reset();
      output_ids.clear();
      std::cout << LLM_GREEN("已重置，请输入新的问题\n");
      continue;
    }
    std::vector<int> input_ids;
    query += "\n";

    model.Tokenize(query, input_ids);

    std::vector<float> next_token_prob;
    next_token_prob.resize(vocab_size, 0.0);
    // process query[:-1]
    for (int i = 0; i < input_ids.size() - 1; ++i) {
      output_ids.emplace_back(input_ids[i]);
      // only update cache
      model.Forward(input_ids[i], &next_token_prob);
    }
    // start decode from query[-1]
    output_ids.emplace_back(input_ids[input_ids.size() - 1]);
    while (output_ids.size() < cache_size) {
      // update cache & next_token
      model.Forward(output_ids[output_ids.size() - 1], &next_token_prob);
      auto it =
          std::max_element(next_token_prob.begin(), next_token_prob.end());
      int next_token = it - next_token_prob.begin();
      output_ids.emplace_back(next_token);
      model.ConvertId2Token(next_token, token);

      if (token == "�") {
        if (prev_token_id == -1) {
          prev_token_id = next_token;
        } else {
          model.ConvertIds2String({prev_token_id, next_token}, token);
          prev_token_id = -1;
        }
      }
      if (token != "�" and token != "\n" && token != "</s>") {
        std::cout << LLM_GREEN(token);
        fflush(stdout);
      }
      if (next_token == 2) {  // eos = 2
        break;
      }
    }
    std::cout << std::endl;
    fflush(stdout);
  }
  return 0;
}
