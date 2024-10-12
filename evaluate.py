# -*- coding: utf-8 -*-
import torch
from loader import load_data

"""
模型效果测试
self.knwb 是一个字典
键是自定义的标准问的index
值是列表，append的是该标准问的编码序号

self.question_index_to_standard_question_index是一个字典
键是从0开始
值是自定义的标准问的index
即数据构成是相似问对应其标准问的index

question_ids 是一个列表
存储的是所有相似问的编码序号

test_question_vectors 是batch_size个样本编码序号解码出来的向量

self.knwb_vectors 是所有相似问的编码序号经过解码出来的向量


测试过程
首先需要清楚的是测试的数据是[相似问的序号，标准问的index]
取出batch_data的相似问的序号 -> 经过model解码出batch_data个向量
然后在batch_data个向量中一个个取出向量，与所有相似问的向量进行矩阵乘法，计算相似度
取出相似度最高的向量的序号，转化成标准问的index，与标准问的index进行比较，如果相同，则认为预测正确，否则认为预测错误。


torch.stack(self.question_ids, dim=0) 是 PyTorch 中用于将一个张量序列沿着指定的维度（这里是第 0 维）进行堆叠操作的一个函数。
具体来说，它会将 self.question_ids 中的张量合并成一个新的张量。
如果 self.question_ids 是一个包含多个形状相同的张量的列表或元组，则此操作将创建一个新的维度并将这些张量沿着这个新的维度排列。

self.question_ids = [torch.tensor([1, 2]), torch.tensor([3, 4])]
使用 torch.stack 进行堆叠：
stacked_tensor = torch.stack(self.question_ids, dim=0)
输出将是：
tensor([[1, 2],
        [3, 4]])

"""

class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)
        self.train_data = load_data(config["train_data_path"], config)
        self.stats_dict = {"correct":0, "wrong":0}  #用于存储测试结果
        
    def knwb_to_vector(self):
        self.question_index_to_standard_question_index = {}
        self.question_ids = []
        for standard_question_index, question_ids in self.train_data.dataset.knwb.items():
            for question_id in question_ids:
                # 记录问题编号到标准问题标号的映射，用来确认答案是否正确
                self.question_index_to_standard_question_index[len(self.question_ids)] = standard_question_index
                self.question_ids.append(question_id)
        with torch.no_grad():
            question_matrixs = torch.stack(self.question_ids, dim=0)
            if torch.cuda.is_available():
                question_matrixs = question_matrixs.cuda()
            self.knwb_vectors = self.model(question_matrixs)
            self.knwb_vectors = torch.nn.functional.normalize(self.knwb_vectors, dim=-1)
        return


    def eval(self, epoch):
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.stats_dict = {"correct":0, "wrong":0}  #清空前一轮的测试结果
        self.model.eval()
        self.knwb_to_vector()
        for index, batch_data in enumerate(self.valid_data):
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            input_ids, labels = batch_data
            with torch.no_grad():
                test_question_vectors = self.model(input_ids)
            self.write_stats(test_question_vectors, labels)
        self.show_stats()
        return


    def write_stats(self, test_question_vectors, labels):
        assert len(test_question_vectors) == len(labels)
        for test_question_vector, label in zip(test_question_vectors, labels):
            # 计算相似度
            similarity = torch.mm(test_question_vector.unsqueeze(0), self.knwb_vectors.T) # torch.mm是PyTorch 中用于矩阵乘法的函数
            hit_index = int(torch.argmax(similarity.squeeze()))
            hit_index = self.question_index_to_standard_question_index[hit_index]
            if int(hit_index) == int(label):
                self.stats_dict["correct"] += 1
            else:
                self.stats_dict["wrong"] += 1
        return


    def show_stats(self):
        correct = self.stats_dict["correct"]
        wrong = self.stats_dict["wrong"]
        self.logger.info("预测集合条目总量：%d" % (correct +wrong))
        self.logger.info("预测正确条目：%d，预测错误条目：%d" % (correct, wrong))
        self.logger.info("预测准确率：%f" % (correct / (correct + wrong)))
        self.logger.info("--------------------")
        return









