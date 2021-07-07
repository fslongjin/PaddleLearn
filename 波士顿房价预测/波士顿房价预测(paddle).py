import paddle
from paddle.nn import Linear
import paddle.nn.functional as F
import numpy as np
import os
import random


def load_data():

    data_file = 'housing.data'
    # 从文件读入数据，并指定分隔符为空格
    # 这里一定要指定dtype为float32！！！不然会报错
    data = np.fromfile(data_file, sep=' ', dtype=np.float32)
    # 此时data.shape为(7084,)

    # 每条数据包含14项，前13项为影响因素，第14项为价格的中位数
    feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
                     'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    feature_num = len(feature_names)

    # 将原始数据进行reshape， 变成[n,14]的形状
    data = data.reshape([data.shape[0] // feature_num, feature_num])
    # 于是，data.shape变成了(506, 14)

    # 将数据集拆分为训练集和测试集
    # 这里使用80%为训练集， 20%为测试集
    # 训练集和测试集必须没有交集

    ratio = 0.8
    offset = int(data.shape[0] * ratio)
    training_data = data[: offset]

    # 计算训练集的最大值、最小值、平均值     形状为（14,）
    maximums = training_data.max(axis=0)
    minimums = training_data.min(axis=0)
    avgs = training_data.sum(axis=0) / training_data.shape[0]

    global max_values
    global min_values
    global avg_values

    max_values = maximums
    min_values = minimums
    avg_values = avgs


    # 对数据进行归一化
    for i in range(feature_num):
        data[:, i] = (data[:, i] - avg_values[i]) / (maximums[i] - minimums[i])

    training_data = data[: offset]
    test_data = data[offset:]
    return training_data, test_data


# 定义模型
class Regressor(paddle.nn.Layer):
    def __init__(self):
        # 实例化父类
        super(Regressor, self).__init__()

        # 定义一层全连接层，输入维度为13，输出维度为1
        self.fc = Linear(in_features=13, out_features=1)

    # 网络的前向计算
    def forward(self, inputs):
        x = self.fc.forward(inputs)
        return x


model = Regressor()


def train():
    # 声明定义好的线性回归模型
    model = Regressor()

    # 开启模型的训练模式
    model.train()
    # 加载数据
    training_data, test_data = load_data()

    # 定义优化算法，使用随机梯度下降SGD
    # 学习率为0.01
    opt = paddle.optimizer.SGD(learning_rate=0.01, parameters=model.parameters())
    # 设置外层循环次数
    EPOCH_NUM = 10
    # 设置batch大小
    BATCH_SIZE = 10

    # 定义外层循环
    for epoch_id in range(EPOCH_NUM):
        # 在每轮迭代开始之前，将训练数据的顺序随机打乱
        np.random.shuffle(training_data)
        # 将训练数据进行拆分，每个batch包含10条数据
        mini_batches = [training_data[k:k + BATCH_SIZE] for k in range(0, len(training_data), BATCH_SIZE)]
        # 定义内层循环
        for iter_id, mini_batch in enumerate(mini_batches):
            # 当前批次训练数据
            x = np.array(mini_batch[:, :-1])
            # 当前批次标签（真实房价）
            y = np.array(mini_batch[:, -1:])

            # 将np数组转换为飞桨动态图tensor
            house_features = paddle.to_tensor(x)
            prices = paddle.to_tensor(y)

            # 前向计算
            predicts = model(house_features)

            # 计算损失
            loss = F.square_error_cost(predicts, label=prices)

            avg_loss = paddle.mean(loss)

            if iter_id % 20 == 0:
                print("epoch:{}, iter:{}, loss:{}".format(epoch_id, iter_id, avg_loss.numpy()))

            # 反向传播
            avg_loss.backward()

            # 最小化loss，更新参数
            opt.step()

            # 清楚梯度
            opt.clear_grad()

    # 保存模型
    paddle.save(model.state_dict(), 'LR_model.pdparams')
    print('模型参数保存成功！')

def load_one_example():
    # 加载数据
    training_data, test_data = load_data()
    idx = np.random.randint(0, test_data.shape[0])
    idx -= 10

    one_data, label = test_data[idx, :-1], test_data[idx, -1:]

    # 修改该条数据为[1, 13]
    one_data = one_data.reshape([1, -1])
    return one_data, label



def predict():
    model_dict = paddle.load('LR_model.pdparams')
    model.load_dict(model_dict)
    model.eval()

    global max_values
    global min_values
    global avg_values

    one_data, label = load_one_example()

    one_data = paddle.to_tensor(one_data)
    predict = model(one_data)
    # 反归一化
    predict = predict * (max_values[-1] - min_values[-1]) + avg_values[-1]
    label = label * (max_values[-1] - min_values[-1]) + avg_values[-1]

    print("Inference result is {}, the corresponding label is {}".format(predict.numpy(), label))



if __name__ == '__main__':
    mode = 'train'
    if mode == 'train':
        train()
    elif mode == 'predict':
        predict()





