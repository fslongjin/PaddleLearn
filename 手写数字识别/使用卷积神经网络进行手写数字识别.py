import paddle
from paddle.nn import Conv2D, MaxPool2D, Linear, Softmax
import paddle.nn.functional as F
import json, gzip
import numpy as np
import random
from PIL import Image

img_height, img_width = 28, 28

# 同步数据读取
def load_data_sync(mode='train'):
    """
    同步读取数据
    """
    data_file = 'work/mnist.json.gz'
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
            img = np.reshape(imgs[i], [1, img_height, img_width]).astype('float32')
            label = np.reshape(labels[i], [1]).astype('int64')
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



# 多层卷积神经网络实现
class MNIST(paddle.nn.Layer):
    def __init__(self):
        super(MNIST, self).__init__()

        # 输出特征通道为20， 卷积核大小为5， 卷积步长为1， padding为2
        self.conv1 = Conv2D(in_channels=1, out_channels=20, kernel_size=5, stride=1, padding=2)
        self.max_pool_1 = MaxPool2D(kernel_size=2, stride=2)

        self.conv2 = Conv2D(in_channels=20, out_channels=20, kernel_size=5, stride=1, padding=2)
        self.max_pool_2 = MaxPool2D(kernel_size=2, stride=2)

        # 定义一层全连接层，输出维度为10
        self.fc = Linear(in_features=980, out_features=10)

    # 定义网络前向计算过程，卷积后紧接着使用池化层，最后使用全连接层计算最终输出
    # 卷积层激活函数使用relu， 全连接层不使用激活函数
    def forward(self, inputs, label=None):
        x = self.conv1(inputs)
        x = F.relu(x)
        x = self.max_pool_1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool_2(x)
        x = paddle.reshape(x, [x.shape[0], -1])
        x = self.fc(x)
        x = F.softmax(x)
        if label is not None:
            # 计算分类准确率
            acc = paddle.metric.accuracy(input=x, label=label)
            return x, acc
        else:
            return x


# 启动训练过程
def train(model):
    model.train()

    print("正在读取数据...")
    # 调用加载数据的函数
    train_loader = load_data_sync('train')

    # 随机梯度下降，每次训练少量数据，抽样偏差导致的参数收敛过程中震荡
    #opt = paddle.optimizer.SGD(learning_rate=0.001, parameters=model.parameters())
    # Momentum： 引入物理“动量”的概念，累积速度，减少震荡，使参数更新的方向更稳定
    #opt = paddle.optimizer.Momentum(learning_rate=0.001, momentum=0.9, parameters=model.parameters())

    # AdaGrad： 根据不同参数距离最优解的远近，动态调整学习率。学习率逐渐下降，依据各参数变化大小调整学习率。
    #opt = paddle.optimizer.Adagrad(learning_rate=0.001, parameters=model.parameters())

    # Adam： 由于动量和自适应学习率两个优化思路是正交的，因此可以将两个思路结合起来，这就是当前广泛应用的算法。
    opt = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())


    EPOCH_NUM = 10

    # MNIST图像的高和宽
    img_height, img_width = 28, 28

    for epoch_id in range(EPOCH_NUM):
        for batch_id, data in enumerate(train_loader()):
            # 准备数据
            images, labels = data
            images = paddle.to_tensor(images)
            # images = paddle.reshape(images, [images.shape[0], 1, img_height, img_width])
            labels = paddle.to_tensor(labels)

            #前向计算
            predicts, acc = model(images, labels)

            # 计算损失，取一个批次样本损失的平均值
            loss = F.cross_entropy(predicts, labels)
            avg_loss = paddle.mean(loss)

            # 每训练200批次的数据，打印当前loss情况
            if batch_id % 200 == 0:
                print("epoch:{}, batch:{}, loss:{}, accuracy:{}".format(epoch_id, batch_id, avg_loss.numpy(), acc.numpy()))

            # 反向传播
            avg_loss.backward()
            opt.step()
            opt.clear_grad()

    paddle.save(model.state_dict(), 'work/mnist-cnn.pdparams')


def load_image(img_path):
    im = Image.open(img_path).convert('L')
    im = im.resize((28, 28), Image.ANTIALIAS)
    im = np.array(im).reshape(1, 1, 28, 28).astype(np.float32)

    # 图像归一化
    im = 1.0 - im/255.

    return im

def predict():
    model = MNIST()
    params_file_path = 'work/mnist-cnn.pdparams'
    img_path = 'work/example_0.png'
    param_dict = paddle.load(params_file_path)
    model.load_dict(param_dict)

    # 预测模式
    model.eval()
    tensor_img = load_image(img_path)
    results = model(paddle.to_tensor(tensor_img))

    # 取概率最大的标签作为预测输出
    lab = np.argsort(results.numpy())

    print("本次预测的数字是：", lab[0][-1])


if __name__ == '__main__':

    mode = 'train'
    if mode == 'train':
        model = MNIST()
        train(model)
    elif mode == 'predict':
        predict()




