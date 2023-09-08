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

#ifndef HOBOT_LLM_INCLUDE_H_
#define HOBOT_LLM_INCLUDE_H_

#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include "model/bpu_bloom_model.h"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

namespace hobot_llm {

class HobotLLMNode : public rclcpp::Node {
 public:
  HobotLLMNode(std::string node_name = "hobot_llm");
  ~HobotLLMNode();

 private:
  void TopicCallback(const std_msgs::msg::String::SharedPtr msg);
  void Run(void);
  void MessageProcess(std::string &query);

  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscription_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;

  llm::BpuBloomModel model_;

  int cache_size_;
  int vocab_size_;

  std::vector<int> output_ids_;

  static constexpr size_t kMaxMessageQueueSize = 10;
  bool thread_exit_;
  std::queue<std_msgs::msg::String::SharedPtr> message_queue_;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::thread thread_;
};

}  // namespace hobot_llm

#endif
