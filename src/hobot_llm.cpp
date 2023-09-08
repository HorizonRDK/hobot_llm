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

#include "hobot_llm.h"

#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"

namespace hobot_llm {

HobotLLMNode::HobotLLMNode(std::string node_name) : Node(node_name) {
  std::string model_dir = "/opt/tros/lib/hobot_llm/llm_model";
  declare_parameter<std::string>("model_dir", model_dir);
  get_parameter<std::string>("model_dir", model_dir);

  std::string tokenizer_dir = "/opt/tros/lib/hobot_llm/tokenization_bloom_py";
  declare_parameter<std::string>("tokenizer_dir", tokenizer_dir);
  get_parameter<std::string>("tokenizer_dir", tokenizer_dir);

  model_.Read(model_dir, tokenizer_dir);
  cache_size_ = model_.GetCacheSize();
  vocab_size_ = model_.GetVocabSize();

  std::string topic_subscription_name = "/text_query";
  declare_parameter<std::string>("topic_sub", topic_subscription_name);
  get_parameter<std::string>("topic_sub", topic_subscription_name);

  subscription_ = create_subscription<std_msgs::msg::String>(
      topic_subscription_name, rclcpp::QoS(10),
      std::bind(&HobotLLMNode::TopicCallback, this, std::placeholders::_1));

  std::string topic_publisher_name = "/text_result";
  declare_parameter<std::string>("topic_pub", topic_publisher_name);
  get_parameter<std::string>("topic_pub", topic_publisher_name);
  publisher_ = create_publisher<std_msgs::msg::String>(topic_publisher_name,
                                                       rclcpp::QoS(10));

  thread_exit_ = false;
  thread_ = std::thread(&HobotLLMNode::Run, this);
}

HobotLLMNode::~HobotLLMNode() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    thread_exit_ = true;
  }
  cv_.notify_all();

  thread_.join();
}

void HobotLLMNode::TopicCallback(const std_msgs::msg::String::SharedPtr msg) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (message_queue_.size() >= kMaxMessageQueueSize) {
    message_queue_.pop();
  }
  message_queue_.push(msg);
  cv_.notify_one();
}

void HobotLLMNode::MessageProcess(std::string &query) {
  RCLCPP_INFO_STREAM(this->get_logger(), "Q: " << query);

  if (output_ids_.size() + 20 >= cache_size_) {
    model_.Reset();
    output_ids_.clear();
  }

  std::vector<int> input_ids;
  query += "\n";  // Add suffix
  model_.Tokenize(query, input_ids);
  std::vector<float> next_token_prob;
  next_token_prob.resize(vocab_size_, 0.0);
  // process query[:-1]
  for (int i = 0; i < input_ids.size() - 1; ++i) {
    output_ids_.emplace_back(input_ids[i]);
    // only update cache
    model_.Forward(input_ids[i], &next_token_prob);
  }
  // start decode from query[-1]
  output_ids_.emplace_back(input_ids[input_ids.size() - 1]);

  auto isPunctuation = [](const std::string &utf8String) {
    auto isChinesePunctuation = [](const std::string &str, size_t index) {
      return (str[index] == '\xEF' && str[index + 1] == '\xBC' &&
              (str[index + 2] == '\x8C' || str[index + 2] == '\x9F' ||
               str[index + 2] == '\x9A' || str[index + 2] == '\x81')) ||
             (str[index] == '\xE3' && str[index + 1] == '\x80' &&
              (str[index + 2] == '\x82' || str[index + 2] == '\x81'));
    };

    for (size_t i = 0; i < utf8String.size();) {
      unsigned char currentByte = utf8String[i];
      if ((currentByte & 0x80) == 0x00) {
        // 单字节字符
        if (std::ispunct(currentByte)) {
          return true;
        }
        i++;
      } else if ((currentByte & 0xF0) == 0xE0) {
        // 三字节字符
        if (isChinesePunctuation(utf8String, i)) {
          return true;
        }
        i += 3;
      } else {
        return false;
      }
    }

    return false;
  };

  std::string string_public;
  int prev_token_id = -1;
  while (output_ids_.size() < cache_size_) {
    // update cache & next_token
    model_.Forward(output_ids_[output_ids_.size() - 1], &next_token_prob);
    auto it = std::max_element(next_token_prob.begin(), next_token_prob.end());
    int next_token = it - next_token_prob.begin();
    output_ids_.emplace_back(next_token);
    std::string token;
    model_.ConvertId2Token(next_token, token);
    if (token == "�") {
      if (prev_token_id == -1) {
        prev_token_id = next_token;
      } else {
        model_.ConvertIds2String({prev_token_id, next_token}, token);
        prev_token_id = -1;
      }
    }
    if (token != "�" && token != "\n" && token != "</s>") {
      string_public += token;
      if (isPunctuation(token)) {
        auto message = std_msgs::msg::String();
        message.data = string_public;
        publisher_->publish(message);
        RCLCPP_INFO_STREAM(this->get_logger(), "Q: " << string_public);
        string_public.clear();
      }
    }
    if (next_token == 2) {  // eos = 2
      break;
    }
  }
}

void HobotLLMNode::Run(void) {
  while (rclcpp::ok()) {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return !message_queue_.empty() || thread_exit_; });

    if (thread_exit_) {
      break;
    }

    auto message = message_queue_.front();
    message_queue_.pop();
    lock.unlock();

    MessageProcess(message->data);
  }
}

}  // namespace hobot_llm
