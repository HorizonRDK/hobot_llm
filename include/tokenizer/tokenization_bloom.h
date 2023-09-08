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


#ifndef TOKENIZATION_BLOOM_H_
#define TOKENIZATION_BLOOM_H_

#include <string>
#include <vector>

#include <Python.h>

namespace llm {

class BloomTokenizer {
 public:
  BloomTokenizer();
  ~BloomTokenizer();

  bool Init(const std::string &py_file_path,
            const std::string &vocab_file_path);
  void Encode(const std::string &query, std::vector<int> &token_ids);
  void Decode(const std::vector<int> &token_ids, std::string &text);
  void Decode(int token_id, std::string &text);

 private:
  PyObject *py_module_;
  PyObject *py_dict_;
  PyObject *py_class_;
  PyObject *py_construct_;
  PyObject *py_instance_;
};

}  // namespace llm

#endif  // TOKENIZATION_BLOOM_H_
