# -*- coding: utf-8 -*-
""" 
@Time    : 2023/5/23 13:02
@Author  : hy
@FileName: utils.py
@SoftWare: PyCharm
"""
import time
from datetime import timedelta
import torch
from tensorboardX import SummaryWriter
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import torch.nn as nn

class config(object):

    """配置参数"""
    def __init__(self):
        self.save_path =  './models/'        # 模型训练结果
        self.log_path =  './logs/'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        self.dropout = 0.1                                             # 随机失活
        self.require_improvement = 50                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_epochs = 100000
        # self.num_epochs = 200
        # epoch数
        self.batch_size = 30000                                           # mini-batch大小
        self.learning_rate = 1e-3                                       # 学习率
        # self.hidden_units = (2048, 1024, 1024, 512, 512, 256, 256, 128, 64, 1)
        # self.hidden_units = (2048, 1024, 512, 256, 128, 64, 1)
        # self.hidden_units = (512, 256, 128, 64, 1)
        # self.hidden_units = (1024, 1024, 512, 64, 1)
        # self.hidden_units = (1024, 1024, 512, 64, 1)
        self.hidden_units = (64, 128, 256, 512, 512, 1024, 1)

        self.activation = 'elu'
        self.dropout_rate = 0
        self.init_std = 0.0001




class configmul(object):
    def __init__(self):
        self.save_path = './models/'  # 模型训练结果
        self.log_path = './logs/'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
        self.dropout = 0.1  # 随机失活
        self.require_improvement = 50  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_epochs = 100000  # epoch数
        self.batch_size = 5000  # mini-batch大小
        self.learning_rate = 1e-3  # 学习率
        self.dropout_rate = 0
        self.init_std = 0.0001
        self.group_hidden_units = [(512, 256,32),(128, 64),(128, 64),(128, 64)]
        self.concat_hidden_units = (512, 256, 1)
        self.group_activation = 'prelu'
        self.concat_activation = 'prelu'
        self.sparse_hidden_units = (128, 64)
        self.embedding_dim = 20

def create_embedding_matrix(X, feature_columns, embedding_dim, init_std=0.0001, linear=False, sparse=False, device='cpu'):

    # Return nn.ModuleDict: for sparse features, {embedding_name: nn.Embedding}
    # for varlen sparse features, {embedding_name: nn.EmbeddingBag}

    d = {}
    for feat in feature_columns:
        d[feat] = nn.Embedding(X[feat].nunique(), embedding_dim if not linear else 1, sparse=sparse)

    embedding_dict = nn.ModuleDict(d)

    for tensor in embedding_dict.values():
        nn.init.normal_(tensor.weight, mean=0, std=init_std)

    return embedding_dict.to(device)

class DatasetIterater(object):
    """
      根据数据集产生batch
      这里需要注意的是，在_to_tensor()中，代码把batch中的数据处理成了`(x, seq_len), y`的形式
      其中x是words_line，seq_len是pad前的长度(超过pad_size的设为pad_size)，y是数据标签
    """
    # 这里的batches就是经过build_dataset()中的load_dataset()处理后得到的contents：(words_line, int(label), seq_len, bigram, trigram)
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size # batch的容量（一次进多少个句子）
        self.batches = batches  # 数据集
        self.n_batches = len(batches) // batch_size # 数据集大小整除batch容量
        self.residue = False  # 记录batch数量是否为整数，false代表可以，true代表不可以，residuere是‘剩余物，残渣'的意思
        if len(batches) % self.batch_size != 0:
            self.residue = True
        self.index = 0 # 迭代用的索引
        self.device = device

    def _to_tensor(self, datas):

        x = torch.tensor(data=datas.iloc[:,0:9].values)
        y = torch.tensor(data=datas.iloc[:,9:10].values)

        return x.float(), y.float()

    def __next__(self):
        if self.residue and self.index == self.n_batches: # 如果batch外还剩下一点句子，并且迭代到了最后一个batch
            batches = self.batches[self.index * self.batch_size: len(self.batches)] # 直接拿出剩下的所有数据
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else: # 迭代器的入口，刚开始self.index是0，肯定小于self.n_batches
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size] # 正常取一个batch的数据
            self.index += 1
            batches = self._to_tensor(batches) # 转化为tensor
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches

def build_iterator(dataset, config):  # 这里的dataset是经过build_dataset()处理后得到的数据（vocab, train, dev, test）
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def train(config, model, train_iter, dev_iter):
    start_time = time.time()
    model.train() # model.train()将启用BatchNormalization和Dropout，相应的，model.eval()则不启用BatchNormalization和Dropout
    # optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate) # 指定优化方法
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)  # 指定优化方法

    # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    criterion = torch.nn.MSELoss()
    writer = SummaryWriter(log_dir=config.log_path + time.strftime('%m-%d_%H.%M', time.localtime()))
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        # scheduler.step() # 学习率衰减
        for i, (trains, labels) in enumerate(train_iter):
            trains = trains.cuda()
            labels = labels.cuda()
            outputs = model(trains)
            model.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if total_batch % 100 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                label_cpu = labels.data.cpu() # 从cpu tensor中取出标签数据
                outputs_cpu = outputs.data.cpu()
                time_dif = get_time_dif(start_time)
                train_rmse = mean_squared_error(label_cpu, outputs_cpu)
                train_r2 = r2_score(label_cpu, outputs_cpu)
                train_loss = loss.item()
                dev_loss, dev_rmse, dev_r2 = evaluate(config, model, dev_iter)  # 计算开发集上的准确率和训练误差
                msg = 'Iter: {0:>6},  train_rmse: {1:>6.2},  train_r2: {2:>6.2},  dev_rmse: {3:>6.2},  dev_r2: {4:>6.2},  Time: {5}, train_loss: {6:>6.2}, dev_loss: {7:>6.2}'
                print(msg.format(total_batch, train_rmse, train_r2, dev_rmse, dev_r2, time_dif, train_loss, dev_loss))
                writer.add_scalar("rmse/train", train_rmse, total_batch)
                writer.add_scalar("rmse/dev", dev_rmse, total_batch)
                writer.add_scalar("r2/train", train_r2, total_batch)
                writer.add_scalar("r2/dev", dev_r2, total_batch)
                model.train()
            total_batch += 1
        if epoch % 1000 == 0:
            torch.save(model.state_dict(), config.save_path + 'ann' + str(epoch) + '.ckpt')
        #     if total_batch - last_improve > config.require_improvement:
        #         # 验证集loss超过1000batch没下降，结束训练
        #         # 开发集loss超过一定数量的batch没下降，则结束训练
        #         print("No optimization for a long time, auto-stopping...")
        #         flag = True
        #         break
        # if flag:
        #     break
    writer.close()
    # test(config, model, test_iter)

def evaluate(config, model, data_iter):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    criterion = torch.nn.MSELoss()
    with torch.no_grad():  # 不追踪梯度
        for texts, labels in data_iter:  # 对数据集中的每一组数据
            texts = texts.cuda()
            outputs = model(texts)  # 使用模型进行预测
            # loss = F.cross_entropy(outputs, labels) # 计算模型损失
            # loss_total += loss # 累加模型损失

            labels = labels.data.cpu().numpy()
            predic = outputs.data.cpu().numpy()
            labels_all = np.append(labels_all, labels) # 记录标签
            predict_all = np.append(predict_all, predic) # 记录预测结果

    loss = criterion(torch.tensor(predict_all[:, np.newaxis]), torch.tensor(labels_all[:, np.newaxis]))
    dev_rmse = mean_squared_error(labels_all, predict_all)
    dev_r2 = r2_score(labels_all, predict_all)

    # if test:
    #     report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
    #     confusion = metrics.confusion_matrix(labels_all, predict_all)
    #     return acc, loss_total / len(data_iter), report, confusion
    return loss, dev_rmse, dev_r2



def test(config, model, test_iter, model_name):
    model.load_state_dict(torch.load(config.save_path + model_name))
    loss, test_rmse, test_r2 = evaluate(config, model, test_iter)
    print(loss, test_rmse, test_r2)

