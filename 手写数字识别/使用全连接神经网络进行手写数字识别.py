import paddle
from paddle.nn import Linear
import paddle.io
from paddle.nn import Layer
import paddle.nn.functional as F
import os
import gzip
import json
import random
import numpy as np


# 同步数据读取
def load_data_sync(mode='train'):
    """
    同步读取数据
    """
    data_file = './mnist.json.gz'
    print("从 {} 读取MNIST数据集...".format(data_file))
    # 加载json数据文件
    data = json.load(gzip.open(data_file))
    print("mnist dataset load done.")

    # 区分训练集验证集和测试集
    train_set, val_set, eval_set = data
    if mode == 'train':
        # 获取训练集
        imgs, labels = train_set[0], train_set[1]
    elif mode == 'valid':
        imgs, labels = val_set[0], val_set[1]
    elif mode == 'eval':
        imgs, labels = eval_set[0], eval_set[1]
    else:
        raise Exception("mode 只有 train, valid, eval三种！")
    print("图像数量：", len(imgs))

    # 校验数据
    assert len(imgs) == len(labels), \
            "imgs数组({} 和labels数组({})的大小应当相同".format(len(imgs), len(labels))

    # 获得数据集长度
    imgs_length = len(imgs)

    # 定义数据集每个数据的序号，根据序号读取数据
    index_list = list(range(imgs_length))
    # 读取数据时用到的批次大小
    BATCHSIZE = 100

    # 定义数据生成器
    def data_generator():
        if mode == 'train':
            # 训练模式下打乱数据
            random.shuffle(index_list)
        imgs_list = []
        labels_list = []

        for i in index_list:
            # 将数据处理成希望的类型
            img = np.array(imgs[i]).astype('float32')
            label = np.array(labels[i]).astype('float32')
            imgs_list.append(img)
            labels_list.append(label)

            if len(imgs_list) == BATCHSIZE:
                # 获得一个batch size 的数据并返回
                yield np.array(imgs_list), np.array(labels_list)

                # 清空数据读取列表
                imgs_list = []
                labels_list = []

        # 将剩余数据返回
        if len(imgs_list) > 0:
            yield np.array(imgs_list), np.array(labels_list)

    return data_generator


class MNIST(paddle.nn.Layer):
    def __init__(self):
        super(MNIST, self).__init__()

        # 定义两层全连接隐含层，输出维度是10， 隐含节点数为10
        self.fc1 = Linear(in_features=784, out_features=10)
        self.fc2 = Linear(in_features=10, out_features=10)
        # 定义一层输出层，输出维度为1
        self.fc3 = Linear(in_features=10, out_features=1)

    # 前向计算函数
    def forward(self, inputs):
        opt1 = self.fc1(inputs)
        opt1 = F.sigmoid(opt1)
        opt2 = self.fc2(opt1)
        opt2 = F.sigmoid(opt2)
        result = self.fc3(opt2)
        return result


# 启动训练过程
def train(model):
    model.train()

    print("正在读取数据...")
    # 调用加载数据的函数
    train_loader = load_data_sync('train')


    opt = paddle.optimizer.SGD(learning_rate=0.001, parameters=model.parameters())

    EPOCH_NUM = 10
    for epoch_id in range(EPOCH_NUM):
        for batch_id, data in enumerate(train_loader()):
            # 准备数据
            images, labels = data
            images = paddle.to_tensor(images)
            labels = paddle.to_tensor(labels)

            #前向计算
            predicts = model(images)

            # 计算损失，取一个批次样本损失的平均值
            loss = F.square_error_cost(predicts, labels)
            avg_loss = paddle.mean(loss)

            # 每训练200批次的数据，打印当前loss情况
            if batch_id % 200 == 0:
                print("epoch:{}, batch:{}, loss:{}".format(epoch_id, batch_id, avg_loss.numpy()))

            # 反向传播
            avg_loss.backward()
            opt.step()
            opt.clear_grad()

    paddle.save(model.state_dict(), './mnist-fc.pdparams')


if __name__ == '__main__':
    model = MNIST()
    train(model)

