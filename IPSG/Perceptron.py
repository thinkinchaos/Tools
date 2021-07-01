import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

if __name__ == "__main__":

    iris = load_iris()  # 导入鸢尾花数据集。有三种花，每种花50个样本，每个样本有4维特征。

    datas = iris.data
    datas = datas[:100]  # 由于我们测试的是二分类，因此只取两种花。
    labels = iris.target
    labels = labels[:100]  # 标签也是只取两种花的。

    datas = datas[:, 0:2]  # 由于我们想可视化出超平面（直线方便可视化），因此特征只取前2维。
    labels[labels == 0] = -1  # 由于感知机的标签是+1和-1，因此把标签0改为-1.（标签为1的不用管）

    # 用API划分数据集。分别是：训练集数据，测试集数据，训练集标签，测试集标签。
    data_train, data_test, label_train, label_test = train_test_split(datas, labels)
    xx = np.linspace(data_test[:, 0].min(), data_test[:, 0].max(), 100)  # 可视化超平面的横坐标是第0维特征值。
    plt.scatter(data_test[:, 0], data_test[:, 1], c=label_test)  # 可视化测试集

    epochs = 1000  # 设置最大周期数（把训练集训练多少遍）。
    l_rate = 0.03  # 设置学习率。
    w = np.random.rand(2)  # w的维度等于特征的维度，为2。初始生成两个数初始化。
    b = np.random.rand(1)  # b是标量。初始化为0。随机生成一个数初始化。
    len_data_train = len(data_train)  # 计算训练集的长度。
    len_data_test = len(data_test)  # 计算训练集的长度。
    for ep in range(epochs):
        loss = 0
        tp_train = 0
        for idx in range(len_data_train):  # 对每个样本进行随机梯度下降法。
            data = data_train[idx]
            label = label_train[idx]
            dw = data * label  # 根据感知机损失函数的数学推导，w的导数刚好是y*x
            db = label  # 根据感知机损失函数的数学推导，b的导数刚好是y
            output = np.dot(data, w) + b  # 计算输出
            if label * output <= 0:  # 根据感知机的原理，只处理分类错误的。
                w += l_rate * dw  # 更新权重
                b += l_rate * db  # 更新偏置
            else:
                tp_train += 1
            loss += (-1 * label * output)
        loss /= len_data_train
        acc_train = tp_train / len_data_train

        tp_test = 0
        for idx in range(len_data_test):  # 对每个样本进行随机梯度下降法。
            data2 = data_test[idx]
            label2 = label_test[idx]
            pred_label = np.sign(np.dot(data2, w) + b)
            if pred_label == label2:
                tp_test += 1
        acc_test = tp_test / len_data_test
        print('epoch:', ep + 1, '\tloss:', np.round(loss, 4).item(), '\ttrain acc:', np.round(acc_train, 3), '\ttest acc:', np.round(acc_test, 3))

    plt.scatter(data_train[:, 0], data_train[:, 1], c=label_train)
    plt.plot(xx, -(w[0] * xx + b) / w[1])
    plt.show()
