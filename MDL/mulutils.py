# -*- coding: utf-8 -*-

import os
import time
from datetime import timedelta
import torch
from tensorboardX import SummaryWriter
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import torch.nn as nn

class configmul(object):
    def __init__(self):
        self.save_path = './models/'
        self.log_path = './logs/'
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')  # 设备
        self.require_improvement = 50
        self.num_epochs = 2800
        self.batch_size = 30000
        self.learning_rate = 1e-3
        self.dropout_rate = 0.05
        self.init_std = 0.0001
        self.group_hidden_units = [(32, 64, 128, 256, 512), (16, 32, 64), (32, 64, 128), (16, 32, 64)]
        self.sparse_hidden_units = (16, 32, 64)
        self.concat_hidden_units = (1024, 1024, 1)
        self.group_activation = 'prelu'
        self.concat_activation = 'prelu'
        self.embedding_dim = 50
        self.l2_reg = 1e-5


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches  # 数据集
        self.n_batches = len(batches) // batch_size
        self.residue = False
        if len(batches) % self.batch_size != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):


        x = torch.tensor(data=datas.iloc[:,0:-1].values)
        y = torch.tensor(data=datas.iloc[:,-1:].values)

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

def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def train(config, model, train_iter, dev_iter):
    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    total_batch = 0
    dev_best_loss = float('inf')
    last_improve = 0
    flag = False
    criterion = torch.nn.MSELoss()
    writer = SummaryWriter(log_dir=config.log_path + time.strftime('%m-%d_%H.%M', time.localtime()))
    for epoch in range(config.num_epochs):

        # scheduler.step() # 学习率衰减
        for i, (trains, labels) in enumerate(train_iter):
            trains = trains.cuda()
            labels = labels.cuda()
            outputs = model(trains)
            model.zero_grad()
            loss = criterion(outputs, labels)
            reg_loss = model.get_regularization_loss()
            total_loss = loss + reg_loss
            total_loss.backward()
            optimizer.step()
            if total_batch % 100 == 0:
                print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
                label_cpu = labels.data.cpu()
                outputs_cpu = outputs.data.cpu()
                time_dif = get_time_dif(start_time)
                train_rmse = np.sqrt(mean_squared_error(label_cpu, outputs_cpu))
                train_mae = mean_absolute_error(label_cpu, outputs_cpu)
                train_r2 = r2_score(label_cpu, outputs_cpu)
                train_loss = loss.item()
                dev_loss, dev_rmse, dev_mae, dev_r2 = evaluate(config, model, dev_iter)

                msg = 'Iter: {0:>6},  train_r2: {1:>6.3},  dev_r2: {2:>6.3} -- train_rmse: {3:>6.3},  dev_rmse: {4:>6.3} -- train_mae: {5:>6.3}, dev_mae: {6:>6.3}'
                print(msg.format(total_batch, train_r2,  dev_r2, train_rmse, dev_rmse, train_mae, dev_mae))

                writer.add_scalars('r2', {'Train': train_r2}, total_batch)
                writer.add_scalars('r2', {'validation': dev_r2}, total_batch)

                writer.add_scalars('rmse', {'Train': train_rmse}, total_batch)
                writer.add_scalars('rmse', {'validation': dev_rmse}, total_batch)

                writer.add_scalars('mae', {'Train': train_mae}, total_batch)
                writer.add_scalars('mae', {'validation': dev_mae}, total_batch)

                model.train()
            total_batch += 1
        if epoch % 100 == 0:
            torch.save(model.state_dict(), config.save_path + 'ann' + str(epoch) + '.ckpt')
        #     if total_batch - last_improve > config.require_improvement:
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
    with torch.no_grad():
        for texts, labels in data_iter:
            texts = texts.cuda()
            outputs = model(texts)
            # loss = F.cross_entropy(outputs, labels)
            # loss_total += loss

            labels = labels.data.cpu().numpy()
            predic = outputs.data.cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    loss = criterion(torch.tensor(predict_all[:, np.newaxis]), torch.tensor(labels_all[:, np.newaxis]))
    dev_rmse = np.sqrt(mean_squared_error(labels_all, predict_all))
    dev_mae = mean_absolute_error(labels_all, predict_all)
    dev_r2 = r2_score(labels_all, predict_all)

    # if test:
    #     report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
    #     confusion = metrics.confusion_matrix(labels_all, predict_all)
    #     return acc, loss_total / len(data_iter), report, confusion
    return loss, dev_rmse, dev_mae, dev_r2


def evaluate_test(config, model, data_iter):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    criterion = torch.nn.MSELoss()
    with torch.no_grad():
        for texts, labels in data_iter:
            texts = texts.cuda()
            outputs = model(texts)
            # loss = F.cross_entropy(outputs, labels)
            # loss_total += loss

            labels = labels.data.cpu().numpy()
            predic = outputs.data.cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    loss = criterion(torch.tensor(predict_all[:, np.newaxis]), torch.tensor(labels_all[:, np.newaxis]))
    dev_rmse = np.sqrt(mean_squared_error(labels_all, predict_all))
    dev_mae = mean_absolute_error(labels_all, predict_all)
    dev_r2 = r2_score(labels_all, predict_all)

    # if test:
    #     report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
    #     confusion = metrics.confusion_matrix(labels_all, predict_all)
    #     return acc, loss_total / len(data_iter), report, confusion
    return labels_all,predict_all,dev_r2,dev_rmse,dev_mae
def test(config, model, test_iter, model_name):
    model.load_state_dict(torch.load(config.save_path + model_name))
    labels_all, predict_all, dev_r2, dev_rmse, dev_mae= evaluate_test(config, model, test_iter)
    return  predict_all, labels_all, dev_r2, dev_rmse, dev_mae


def test1(config, model, test_iter, model_name):
    model.load_state_dict(torch.load(config.save_path + model_name))
    labels_all, predict_all, dev_r2, dev_rmse, dev_mae= evaluate_test(config, model, test_iter)
    return  predict_all