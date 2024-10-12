# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import random
import jieba
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
"""
数据加载
"""


class MyDataset:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.schema = load_schema(config["schema_path"])
        self.train_data = config["epoch_data_size"] #由于采取随机采样，所以需要设定一个采样数量，否则可以一直采
        self.data_type = None
        self.load()

    def load(self):
        self.data = []
        self.knwb = defaultdict(list) # 字典，值是列表
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = json.loads(line) # 用于从字符串中读取 JSON 数据，比如 json_string = '{"name": "Alice", "age": 30}'
                # 加载训练集
                if isinstance(line, dict):
                    self.data_type = "train"
                    questions = line["questions"]
                    target_question = line["target"]
                    label = self .schema[target_question]
                    for question in questions:
                        input_id = self.encode_sentence(question)
                        input_id = torch.LongTensor(input_id)
                        self.knwb[label].append(input_id)
                else:
                    self.data_type = "test"
                    assert isinstance(line, list)
                    question, label = line
                    input_id = self.encode_sentence(question)
                    input_id = torch.LongTensor(input_id)
                    label_index = torch.LongTensor([self.schema[label]])
                    self.data.append([input_id, label_index])
        return 

    def encode_sentence(self, sentence):
        input_id = []
        if self.config["vocab_path"] == "words.txt":
            for word in jieba.cut(sentence):
                input_id.append(self.vocab.get(word, self.vocab["[UNK]"]))
        else:
            for char in sentence:
                input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        input_id = self.padding(input_id)
        return input_id

    # 补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id):
        input_id = input_id[:self.config["max_length"]]
        input_id += [0] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        if self.data_type == "train":
            return self.config["epoch_data_size"]
        else:
            assert self.data_type == "test", self.data_type
            return len(self.data)

    def __getitem__(self, index):
        if self.data_type == "train":
            return self.random_train_sample() #随机生成一个训练样本
        else:
            return self.data[index]
                
    """
    依照一定概率生成负样本或正样本
    负样本从随机两个不同的标准问题中各随机选取一个
    正样本从随机一个标准问题中随机选取两个
    正样本的label为1, 负样本的label为-1
    """
    def random_train_sample(self):
        standard_question_index = list(self.knwb.keys())
        # 随机正样本
        if random.random() < 0.5:
            p = random.choice(standard_question_index)
            if len(self.knwb[p]) < 2:
                return self.random_train_sample()
            else:
                s1, s2 = random.sample(self.knwb[p], 2)
                return [s1, s2, torch.LongTensor([1])]
        # 随机负样本
        else:
            p, n = random.sample(standard_question_index, 2)
            s1 = random.choice(self.knwb[p])
            s2 = random.choice(self.knwb[n])
            return [s1, s2, torch.LongTensor([-1])]


# 加载schema
def load_schema(schema_path):
    with open(schema_path, encoding="utf8") as f:
        return json.loads(f.read())


# 加载字表或词表
def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf-8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1
    return token_dict
    

# 用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dataset = MyDataset(data_path, config)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=shuffle)
    return dataloader


if __name__ == "__main__":
    from config import Config
    dataset = MyDataset("./data/train.json", Config)




