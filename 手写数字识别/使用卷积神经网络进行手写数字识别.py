import paddle
from paddle.nn import Conv2D, MaxPool2D, Linear, Softmax
import paddle.nn.functional as F
import json, gzip
import numpy as np
import random
from PIL import Image

from visualdl import LogWriter
# 设置日志保存路径
log_writer = LogWriter("./work/log")

import matplotlib.pyplot as plt



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
    # @paddle.jit.to_static # 添加装饰器，使动态图网络结构在静态图模式下运行
    def forward(self, inputs, label=None, check_shape=False, check_content=False):
        # 给不同层的输出不同命名，方便调试
        outputs1 = self.conv1(inputs)
        outputs2 = F.relu(outputs1)
        outputs3 = self.max_pool_1(outputs2)
        outputs4 = self.conv2(outputs3)
        outputs5 = F.relu(outputs4)
        outputs6 = self.max_pool_2(outputs5)
        outputs6 = paddle.reshape(outputs6, [outputs6.shape[0], -1])
        outputs7 = self.fc(outputs6)
        outputs8 = F.softmax(outputs7)

        # 选择是否打印神经网络每层的参数尺寸和输出尺寸，验证网络结构是否正确
        if check_shape:
            # 打印每层网络设置的超参数-卷积核尺寸，卷积步长，卷积padding，池化核尺寸
            print("\n########## print network layer's superparams ##############")
            print("conv1-- kernel_size: {}, padding: {}, stride: {}".format(self.conv1.weight.shape,
                                                                            self.conv1._padding, self.conv1._stride))
            print("conv2-- kernel_size: {}, padding: {}, stride: {}".format(self.conv2.weight.shape,
                                                                            self.conv2._padding, self.conv2._stride))
            print("fc-- weight_size:{}, bias_size_{}".format(self.fc.weight.shape, self.fc.bias.shape))

            # 打印每层的输出尺寸
            print("\n########## print shape of features of every layer ###############")
            print("inputs_shape: {}".format(inputs.shape))
            print("outputs1_shape: {}".format(outputs1.shape))
            print("outputs2_shape: {}".format(outputs2.shape))
            print("outputs3_shape: {}".format(outputs3.shape))
            print("outputs4_shape: {}".format(outputs4.shape))
            print("outputs5_shape: {}".format(outputs5.shape))
            print("outputs6_shape: {}".format(outputs6.shape))
            print("outputs7_shape: {}".format(outputs7.shape))
            print("outputs8_shape: {}".format(outputs8.shape))

        # 选择是否打印训练过程中的参数和输出内容，可用于训练过程中的调试
        if check_content:
            # 打印卷积层的参数-卷积核权重，权重参数较多，此处只打印部分参数
            print("\n########## print convolution layer's kernel ###############")
            print("conv1 params -- kernel weights:", self.conv1.weight[0][0])
            print("conv2 params -- kernel weights:", self.conv2.weight[0][0])

            # 创建随机数，随机打印某一个通道的输出值
            idx1 = np.random.randint(0, outputs1.shape[1])
            idx2 = np.random.randint(0, outputs4.shape[1])
            # 打印卷积-池化后的结果，仅打印batch中第一个图像对应的特征
            print("\nThe {}th channel of conv1 layer: ".format(idx1), outputs1[0][idx1])
            print("The {}th channel of conv2 layer: ".format(idx2), outputs4[0][idx2])
            print("The output of last layer:", outputs8[0], '\n')

        if label is not None:
            # 计算分类准确率
            acc = paddle.metric.accuracy(input=outputs8, label=label)
            return outputs8, acc
        else:
            return outputs8


# 启动训练过程
def train(model, ckpt=False):
    model.train()

    print("正在读取数据...")
    # 调用加载数据的函数
    train_loader = load_data_sync('train')

    if ckpt:
        # 断点恢复
        params_dict = paddle.load('./work/checkpoints/' + checkpoint_name + ".pdparams")
        opt_dict = paddle.load('./work/checkpoints/' + checkpoint_name + ".pdopt")
        model.set_state_dict(params_dict)

    # 随机梯度下降，每次训练少量数据，抽样偏差导致的参数收敛过程中震荡
    #opt = paddle.optimizer.SGD(learning_rate=0.001, parameters=model.parameters())
    # Momentum： 引入物理“动量”的概念，累积速度，减少震荡，使参数更新的方向更稳定
    #opt = paddle.optimizer.Momentum(learning_rate=0.001, momentum=0.9, parameters=model.parameters())

    # AdaGrad： 根据不同参数距离最优解的远近，动态调整学习率。学习率逐渐下降，依据各参数变化大小调整学习率。
    #opt = paddle.optimizer.Adagrad(learning_rate=0.001, parameters=model.parameters())

    # Adam： 由于动量和自适应学习率两个优化思路是正交的，因此可以将两个思路结合起来，这就是当前广泛应用的算法。
    # weight_decay引入正则化项，coeff调整正则化项的权重。
    opt = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters(), weight_decay=paddle.regularizer.L2Decay(coeff=0.001))

    if ckpt:
        opt.set_state_dict(opt_dict)


    EPOCH_NUM = 10

    # MNIST图像的高和宽
    img_height, img_width = 28, 28

    iters = []
    losses = []
    iter = 0

    for epoch_id in range(start_epoch, EPOCH_NUM):
        for batch_id, data in enumerate(train_loader()):
            # 准备数据
            images, labels = data
            images = paddle.to_tensor(images)
            # images = paddle.reshape(images, [images.shape[0], 1, img_height, img_width])
            labels = paddle.to_tensor(labels)

            """
            # 前向计算，同时得到模型输出值和分类准确率
            if batch_id == 0 and epoch_id == 0:
                predicts, acc = model(images, labels, check_shape=True, check_content=False)
            elif batch_id == 401:
                predicts, acc = model(images, labels, check_shape=False, check_content=True)
            else:
                predicts, acc = model(images, labels)
            """
            predicts, acc = model(images, labels)
            # 计算损失，取一个批次样本损失的平均值
            loss = F.cross_entropy(predicts, labels)
            avg_loss = paddle.mean(loss)

            # 每训练100批次的数据，打印当前loss情况
            if batch_id % 100 == 0:
                print("epoch:{}, batch:{}, loss:{}, accuracy:{}".format(epoch_id, batch_id, avg_loss.numpy(), acc.numpy()))
                iters.append(iter)
                losses.append(avg_loss.numpy())

                # 使用visual DL进行绘图
                log_writer.add_scalar(tag='acc', step=iter, value=acc.numpy())
                log_writer.add_scalar(tag='loss', step=iter, value=avg_loss.numpy())
                # 输出在测试集中的准确率
                acc_val_mean = evaluation(model, False)
                log_writer.add_scalar(tag='eval_acc', step=iter, value=acc_val_mean)

                iter += 100
            # 反向传播
            avg_loss.backward()
            opt.step()
            opt.clear_grad()
        # 保存模型参数和优化器的参数
        paddle.save(model.state_dict(), './work/checkpoints/mnist_epoch{}'.format(epoch_id) + '.pdparams')
        paddle.save(opt.state_dict(), './work/checkpoints/mnist_epoch{}'.format(epoch_id) + '.pdopt')

    paddle.save(model.state_dict(), 'work/mnist-cnn.pdparams')
    """
    
    # 保存静态图网络模型
    from paddle.static import InputSpec
    paddle.jit.save(
        layer=model,
        path='./work/static_graph/mnist',
        input_spec=[InputSpec(shape=[None, 784], dtype='float32')]
    )
    print("静态图成功保存！")
    """
    # 画图
    plt.figure()
    plt.title("train loss", fontsize=24)
    plt.xlabel("iter", fontsize=14)
    plt.ylabel("loss", fontsize=14)
    plt.plot(iters, losses, color="red", label="train loss")
    plt.grid()

    plt.savefig("work/train_loss.png")
    plt.show()


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


eval_loader = load_data_sync('eval')


def evaluation(model, is_empty_model=True):
    print("start evaluation...")

    if is_empty_model:
        # 定义预测过程
        params_file_path = "work/mnist-cnn.pdparams"
        # 加载模型参数
        param_dict = paddle.load(params_file_path)
        model.load_dict(param_dict)

    model.eval()


    acc_set = []
    avg_loss_set = []

    for batch_id, data in enumerate(eval_loader()):
        images, labels = data
        images = paddle.to_tensor(images)
        labels = paddle.to_tensor(labels)

        predicts, acc = model(images, labels)
        loss = F.cross_entropy(input=predicts, label=labels)
        avg_loss = paddle.mean(loss)
        acc_set.append(float(acc.numpy()))
        avg_loss_set.append(float(avg_loss.numpy()))

    # 计算多个batch的平均损失和准确率
    acc_val_mean = np.array(acc_set).mean()
    avg_loss_mean = np.array(avg_loss_set).mean()


    print("loss={}, acc={}".format(avg_loss_mean, acc_val_mean))
    return acc_val_mean




if __name__ == '__main__':

    mode = 'train'

    import os
    if not os.path.exists("./work/checkpoints"):
        os.mkdir("./work/checkpoints")
    global checkpoint_name
    global start_epoch
    start_epoch = 0

    if mode == 'train':
        model = MNIST()
        train(model)
    elif mode == 'predict':
        predict()
    elif mode == 'eval':
        model = MNIST()
        evaluation(model)
    elif mode == 'continue_train':
        model = MNIST()
        checkpoint_name = "mnist_epoch1"
        start_epoch = 1
        train(model, True)





