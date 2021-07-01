import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import torch
import math

if __name__ == "__main__":

    iris = load_iris()

    datas = iris.data

    labels = iris.target

    data_train, data_test, label_train, label_test = train_test_split(datas, labels)


    def feature_normalize(data):
        me = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        return (data - me) / std


    data_train = feature_normalize(data_train)
    data_test = feature_normalize(data_test)

    batch_size, feature_dim = data_train.shape
    x = torch.from_numpy(data_train).float()
    label_train = torch.from_numpy(label_train).unsqueeze(dim=1).long()
    y_onehot = torch.zeros((len(data_train), 3)).scatter_(1, label_train, 1)

    data_test = torch.from_numpy(data_test).float()
    label_test = torch.from_numpy(label_test).unsqueeze(dim=1).long()
    label_test_one_hot = torch.zeros((len(data_test), 3)).scatter_(1, label_test, 1)

    hidden_layer_fea_num = 3
    w1 = torch.ones(feature_dim, hidden_layer_fea_num)
    w2 = torch.ones(hidden_layer_fea_num, 3)

    epochs = 1000
    lr = 1e-4
    for ep in range(epochs):
        hidden_layer = x.mm(w1).clamp(min=0)
        y_predict = hidden_layer.mm(w2)
        loss = (y_predict - y_onehot).pow(2).sum() / batch_size
        print('Epoch:{}, Loss:{}'.format(ep + 1, loss.item()))

        d_w2 = hidden_layer.t().mm(2 * (y_predict - y_onehot))
        d_hidden = 2 * (y_predict - y_onehot).mm(w2.t()).clamp(min=0)
        d_w1 = x.t().mm(d_hidden)

        w1 -= lr * d_w1
        w2 -= lr * d_w2

    y_predict_test = data_test.mm(w1).clamp(min=0).mm(w2).clamp(min=0)
    y_predict_label = torch.max(y_predict_test, 1)[1]
    tp = 0
    for i, item in enumerate(y_predict_label):
        print('Predicted label of the {}th sample is {}, while its ground truth is {}.'.format(i, item, label_test[i][0]))
        if item == label_test[i]:
            tp += 1
    print('Accuracy is:', tp / len(y_predict_label))
