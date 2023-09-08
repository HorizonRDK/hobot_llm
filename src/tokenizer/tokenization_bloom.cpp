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

#include "tokenizer/tokenization_bloom.h"

#include <Python.h>

#include <iostream>
#include <sstream>

namespace llm {

static const char kPythonFileName[] = "tokenization_bloom";
static const char kPythonClassName[] = "BloomTokenizer";
static const char kPythonEncodeFunctionName[] = "tokenize";
static const char kPythonDecodeFunctionName[] = "decode";

BloomTokenizer::BloomTokenizer()
    : py_module_(nullptr),
      py_dict_(nullptr),
      py_class_(nullptr),
      py_construct_(nullptr),
      py_instance_(nullptr) {}

BloomTokenizer::~BloomTokenizer() {
  if (Py_IsInitialized()) {
    if (py_module_) Py_DECREF(py_module_);
    if (py_dict_) Py_DECREF(py_dict_);
    if (py_class_) Py_DECREF(py_class_);
    if (py_construct_) Py_DECREF(py_construct_);
    if (py_instance_) Py_DECREF(py_instance_);
    Py_Finalize();
  }
}

bool BloomTokenizer::Init(const std::string &py_file_path,
                          const std::string &vocab_file_path) {
  if (py_file_path.empty() || vocab_file_path.empty()) return false;
  // Initialize Python interface
  Py_Initialize();
  if (Py_IsInitialized() == 0) {
    std::cout << "Fail to initialize python!" << std::endl;
    return false;
  }

  // Initialize Python file path
  int ret = 0;
  ret = PyRun_SimpleString("import sys");
  if (ret != 0) {
    Py_Finalize();
    return false;
  }
  std::stringstream str_py_cmd;
  str_py_cmd << "sys.path.append('" << py_file_path << "')";
  ret = PyRun_SimpleString(str_py_cmd.str().c_str());
  if (ret != 0) {
    Py_Finalize();
    return false;
  }

  // Import python module
  py_module_ = PyImport_ImportModule(kPythonFileName);
  if (py_module_ == nullptr) {
    PyErr_Print();
    std::cout << "Can't find the python file!" << std::endl;
    Py_DECREF(py_module_);
    Py_Finalize();
    return false;
  }

  // Load the class and method in the python module
  py_dict_ = PyModule_GetDict(py_module_);
  if (py_dict_ == nullptr) {
    std::cout << "Can't find the dictionary!" << std::endl;
    Py_DECREF(py_dict_);
    Py_DECREF(py_module_);
    Py_Finalize();
    return false;
  }

  // Get BloomTokenizer class
  py_class_ = PyDict_GetItemString(py_dict_, kPythonClassName);
  if (py_class_ == nullptr) {
    std::cout << "Can't find BloomTokenizer class!" << std::endl;
    Py_DECREF(py_class_);
    Py_DECREF(py_dict_);
    Py_DECREF(py_module_);
    Py_Finalize();
    return false;
  }

  // Get the constructor of the class
  py_construct_ = PyInstanceMethod_New(py_class_);
  if (py_construct_ == nullptr) {
    std::cout << "Can't create BloomTokenizer class!" << std::endl;
    Py_DECREF(py_construct_);
    Py_DECREF(py_class_);
    Py_DECREF(py_dict_);
    Py_DECREF(py_module_);
    Py_Finalize();
    return false;
  }

  // Create an instance of the BloomTokenizer class
  PyObject *py_init_args = PyTuple_New(1);
  PyObject *py_vocab_file_path = Py_BuildValue("s", vocab_file_path.c_str());
  PyTuple_SetItem(py_init_args, 0, py_vocab_file_path);
  py_instance_ = PyObject_CallObject(py_construct_, py_init_args);
  Py_DECREF(py_vocab_file_path);
  Py_DECREF(py_init_args);
  if (py_instance_ == nullptr) {
    std::cout << "Can't create BloomTokenizer instance!" << std::endl;
    Py_DECREF(py_instance_);
    Py_DECREF(py_construct_);
    Py_DECREF(py_class_);
    Py_DECREF(py_dict_);
    Py_DECREF(py_module_);
    Py_Finalize();
    return false;
  }

  return true;
}

void BloomTokenizer::Encode(const std::string &query,
                            std::vector<int> &token_ids) {
  token_ids.clear();
  PyObject *py_return = PyObject_CallMethod(
      py_instance_, kPythonEncodeFunctionName, "s", query.c_str());
  int item_value;
  if (PyList_Check(py_return)) {
    int pSize = PyList_Size(py_return);
    token_ids.reserve(pSize);
    for (int i = 0; i < pSize; ++i) {
      PyObject *item = PyList_GetItem(py_return, i);
      PyArg_Parse(item, "i", &item_value);
      token_ids.push_back(item_value);
    }
  }
  Py_DECREF(py_return);
  return;
}

void BloomTokenizer::Decode(const std::vector<int> &token_ids,
                            std::string &text) {
  text.clear();
  if (token_ids.empty()) {
    return;
  }
  std::stringstream str_token_ids;
  for (auto i = 0; i < token_ids.size(); ++i) {
    if (i != 0) {
      str_token_ids << ";";
    }
    str_token_ids << std::to_string(token_ids[i]);
  }
  PyObject *py_return =
      PyObject_CallMethod(py_instance_, kPythonDecodeFunctionName, "s",
                          str_token_ids.str().c_str());
  if (nullptr != py_return) {
    text = PyUnicode_AsUTF8(py_return);
  }
  Py_DECREF(py_return);
  return;
}

void BloomTokenizer::Decode(int token_id, std::string &text) {
  text.clear();
  PyObject *py_return = PyObject_CallMethod(
      py_instance_, kPythonDecodeFunctionName, "i", token_id);
  if (nullptr != py_return) {
    text = PyUnicode_AsUTF8(py_return);
  }
  Py_DECREF(py_return);
  return;
}

}  // namespace llm
