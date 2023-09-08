#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from transformers import BloomTokenizerFast


class BloomTokenizer:
    def __init__(self, vocab_file):
        self.tokenizer = BloomTokenizerFast.from_pretrained(vocab_file)

    def tokenize(self, query):
        input_ids = self.tokenizer.encode(query)
        return input_ids

    def decode(self, input):
        text = ""
        if isinstance(input, str) and input:
            input_ids = input.strip().split(";")
            input_ids = [int(i) for i in input_ids]
            text = self.tokenizer.decode(input_ids)
        elif isinstance(input, int):
            if input < 0:
                return text
            text = self.tokenizer.decode(input)
        return text


if __name__ == "__main__":
    vocab_file = "./bloom_1b4_zh"
    bt = BloomTokenizer(vocab_file)
    query = "杂申椒与菌桂兮，岂维纫夫蕙茝"
    print("query: {}".format(query))
    input_ids = bt.tokenize(query)
    print("input_ids: {}".format(input_ids))
    input_ids = [str(s) for s in input_ids]
    input_ids_str = ";".join(input_ids)
    output_str = bt.decode(input_ids_str)
    print("output_str: {}".format(output_str))
