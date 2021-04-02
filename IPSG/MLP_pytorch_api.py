import torch
from torch.autograd import Variable
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

    x = torch.from_numpy(data_train).float().requires_grad_()
    y = torch.from_numpy(label_train).unsqueeze(dim=1).long()
    y_onehot = torch.zeros((batch_size, 3)).scatter_(1, y, 1).requires_grad_()

    x1 = torch.from_numpy(data_test).float()
    y1 = torch.from_numpy(label_test).unsqueeze(dim=1).long()
    y1_onehot = torch.zeros((len(data_test), 3)).scatter_(1, y1, 1)

    hidden_feature_dim = 3

    MLP = torch.nn.Sequential(torch.nn.Linear(feature_dim, hidden_feature_dim),
                              torch.nn.ReLU(),
                              torch.nn.Linear(hidden_feature_dim, 3))

    epochs = 1000
    lr = 1e-3
    loss_function = torch.nn.MSELoss()
    # loss_function = torch.nn.CrossEntropyLoss()
    optimzer = torch.optim.Adam(MLP.parameters(), lr=lr)
    for ep in range(epochs):
        y_pred = MLP(x)
        loss = loss_function(y_pred, y_onehot)
        # print(y_pred.dtype, y_onehot.dtype)
        print('Epoch:{}, Loss:{}'.format(ep + 1, loss.item()))
        optimzer.zero_grad()
        loss.backward()
        optimzer.step()

    tp = 0
    with torch.no_grad():
        y1_pred = MLP(x1)
        y1_pred = torch.max(y1_pred, 1)[1]
        for i, item in enumerate(y1_pred):
            print('Predicted label of the {}th sample is {}, while its ground truth is {}.'.format(i, item, label_test[i]))
            if item == label_test[i]:
                tp += 1
        print('Accuracy is:', tp / len(data_test))
