import cv2
import random
import numpy as np
import os
import paddle
from paddle.nn import Conv2D, MaxPool2D, Linear, Dropout, BatchNorm2D, Softmax
import paddle.nn.functional as F
from visualdl import LogWriter
# 设置日志保存路径
log_writer = LogWriter("./work/log")

DATADIR = 'work/PALM-Training400/PALM-Training400'
DATADIR2 = 'work/PALM-Validation400'
CSVFILE = 'work/labels.csv'


# 对读入的图像进行预处理
def transform_img(img):
    img = cv2.resize(img, (224, 224))
    # 读入图像的数据格式是[H,W,C]
    # 使用转置操作使其变成[C,H,W]
    img = np.transpose(img, (2, 0, 1)).astype('float32')
    # 将数据范围调整到[-1.0, 1.0]之间
    img = img/255.
    img = img*2.0 - 1.0
    return img


# 定义训练集数据读取器
def data_loader(datadir, batchsize=10, mode='train'):
    filenames = os.listdir(datadir)
    def reader():
        if mode == 'train':
            # 训练时数据乱序
            random.shuffle(filenames)
        batch_imgs = []
        batch_labels = []
        for name in filenames:
            filepath = os.path.join(datadir, name)
            img = cv2.imread(filepath)
            img = transform_img(img)

            if name[0] == 'H' or name[0] == 'N':
                # H开头的文件名表示高度近似，N开头的文件名表示正常视力
                # 高度近视和正常视力的样本，都不是病理性的，属于负样本，标签为0
                label = 0
            elif name[0] == 'P':
                # P开头的是病理性近视，属于正样本，标签为1
                label = 1
            else:
                raise ('Not support file name')

            label = np.reshape(label, [1])

            # 每读取一个样本的数据，就将其放入数据列表中
            batch_imgs.append(img)
            batch_labels.append(label)

            if len(batch_imgs) == batchsize:
                imgs_array = np.array(batch_imgs).astype('float32')
                labels_array = np.array(batch_labels).astype('int64')
                yield imgs_array, labels_array
                batch_imgs = []
                batch_labels = []

        if len(batch_imgs) > 0:
            imgs_array = np.array(batch_imgs).astype('float32')
            labels_array = np.array(batch_labels).astype('int64')
            yield imgs_array, labels_array
    return reader


# 定义验证集数据读取器
def valid_data_loader(datadir, csvfile, batch_size=10, mode='valid'):
    # 训练集读取时通过文件名来确定样本标签，验证集则通过csvfile来读取每个图片对应的标签
    # 请查看解压后的验证集标签数据，观察csvfile文件里面所包含的内容
    # csvfile文件所包含的内容格式如下，每一行代表一个样本，
    # 其中第一列是图片id，第二列是文件名，第三列是图片标签，
    # 第四列和第五列是Fovea的坐标，与分类任务无关
    # ID,imgName,Label,Fovea_X,Fovea_Y
    # 1,V0001.jpg,0,1157.74,1019.87
    # 2,V0002.jpg,1,1285.82,1080.47

    # 打开包含验证集标签的csvfile，并读入其中的内容
    file_lists = open(csvfile).readlines()

    def reader():
        batch_imgs = []
        batch_labels = []
        for line in file_lists[1:]:
            line = line.strip().split(',')
            name = line[1]
            # print(line)
            label = int(line[2])
            label = np.reshape(label, [1])
            # 根据图片文件名加载图片，并对图像数据作预处理
            file_path = os.path.join(datadir, name)
            img = cv2.imread(file_path)
            img = transform_img(img)
            # 每读取一个样本的数据，就将其放入数据列表中
            batch_imgs.append(img)
            batch_labels.append(label)

            if len(batch_imgs) == batch_size:
                imgs_array = np.array(batch_imgs).astype('float32')
                labels_array = np.array(batch_labels).astype('int64')
                yield imgs_array, labels_array
                # 清空数据读取列表
                batch_imgs = []
                batch_labels = []


        if len(batch_imgs) > 0:
            imgs_array = np.array(batch_imgs).astype('float32')
            labels_array = np.array(batch_labels).astype('int64')
            yield imgs_array, labels_array
    return reader


# ResNet中使用了BatchNorm层，在卷积层的后面加上BatchNorm以提升数值稳定性
# 定义卷积归一化块
class ConvBNLayer(paddle.nn.Layer):
    def __init__(self,
                 num_channels, num_filters, filter_size, stride=1, groups=1, act=None):
        """
        num_channels 卷积层的输入通道数
        num_filters 卷积层的输出通道数
        stride 卷积层的步幅
        groups 分组卷积的组数， 默认groups=1不使用分组卷积
        """
        super(ConvBNLayer, self).__init__()

        self._conv = Conv2D(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            bias_attr=False)

        self._batch_norm = BatchNorm2D(num_filters)
        self.act = act

    def forward(self, x):
        y = self._conv(x)
        y = self._batch_norm(y)
        if self.act == 'leaky':
            y = F.leaky_relu(x=y, negative_slope=0.1)
        elif self.act == 'relu':
            y = F.relu(x=y)

        return y


# 定义残差块
# 每个残差块会对输入图片进行三次卷积，然后跟输入图片进行短接
# 如果残差块中第三次卷积输出特征图形状与输入不一致，则对图片进行1*1卷积，将其输出形状调整成一致
class BottleneckBlock(paddle.nn.Layer):
    def __init__(self, num_channels, num_filters, stride, shortcut=True):
        super(BottleneckBlock, self).__init__()
        # 创建第一个卷积层 1*1
        self.conv0 = ConvBNLayer(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=1,
            act='relu'
        )
        # 创建第二个卷积层3*3
        self.conv1 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            act='relu'
        )
        # 创建第三个卷积1*1，但是输出通道乘4
        self.conv2 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters*4,
            filter_size=1,
            act=None
        )

        # 如果conv2的输出跟此残差块的输入数据形状一致，则shortcut=True
        # 否则shortcut = False，添加1个1x1的卷积作用在输入数据上，使其形状变成跟conv2一致
        if not shortcut:
            self.short = ConvBNLayer(
                num_channels=num_channels,
                num_filters=num_filters*4,
                filter_size=1,
                stride=stride
            )

        self.shortcut = shortcut
        self._num_channels_out = num_filters*4

    def forward(self, x):
        y = self.conv0(x)
        conv1 = self.conv1(y)
        conv2 = self.conv2(conv1)

        # 如果shortcut=True， 直接将inputs和conv2的输出相加
        # 否则需要对inputs进行一次卷积，将形状调整成跟conv2一致

        if self.shortcut:
            short = x
        else:
            short = self.short(x)

        y = paddle.add(short, conv2)
        y = F.relu(y)

        return y


# 定义ResNet模型
class ResNet(paddle.nn.Layer):
    def __init__(self, layers=50, class_dim=1):
        """
            layers 网络层数，可以是50， 101或者152
            class_dim, 分类的类别数
        """
        super(ResNet, self).__init__()
        self.layers = layers
        supported_layers = [50, 101, 152]
        assert layers in supported_layers, \
            "supported layers are {} but input layer is {}".format(supported_layers, layers)

        if layers == 50:
            # ResNet50包含多个模块，其中第2到5个模块分别包含3，4，6，3个残差块
            depth = [3, 4, 6, 3]
        elif layers == 101:
            # ResNet101包含多个模块，其中第2到5个模块分别包含3，4，23，3个残差块
            depth = [3, 4, 23, 3]
        elif layers == 152:
            # ResNet152包含多个模块，其中第2到5个模块分别包含3，8，36，3个残差块
            depth = [3, 8, 36, 3]

        # 残差块中用到的卷积的输出通道数
        num_filters = [64, 128, 256, 512]

        # ResNet的第一个模块， 包含1个7*7卷积，后面紧跟一个最大池化层
        self.conv = ConvBNLayer(
            num_channels=3,
            num_filters=64,
            filter_size=7,
            stride=2,
            act='relu'
        )

        self.pool2d_max = MaxPool2D(
            kernel_size=3,
            stride=2,
            padding=1
        )

        # ResNet的第二到第五个模块c2, c3, c4, c5
        self.bottleneck_block_list = []
        num_channels = 64
        for block in range(len(depth)):
            shortcut = False
            for i in range(depth[block]):
                bottleneck_block = self.add_sublayer(
                    'bb_%d_%d' % (block, i),
                    BottleneckBlock(
                        num_channels=num_channels,
                        num_filters=num_filters[block],
                        # c3、c4、c5将会在第一个残差块使用stride=2；其余所有残差块stride=1
                        stride=2 if i == 0 and block != 0 else 1,
                        shortcut=shortcut))

                num_channels = bottleneck_block._num_channels_out
                self.bottleneck_block_list .append(bottleneck_block)
                shortcut = True

        # 在c5的输出特征图上使用全局池化
        self.pool2d_avg = paddle.nn.AdaptiveAvgPool2D(output_size=1)

        # stdv用来作为全连接层随机初始化参数的方差
        import math
        stdv = 1.0 / math.sqrt(2048 * 1.0)

        # 创建全连接层，输出大小为类别数目，经过残差网络的卷积和全局池化后，
        # 卷积的特征维度为[B, 2048, 1, 1]故最后一层全连接的输入维度是2048
        self.out = Linear(in_features=2048, out_features=class_dim,
                          weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.Uniform(-stdv, stdv)))

    def forward(self, x, label=None):
        y = self.conv(x)
        y = self.pool2d_max(y)
        for bottlenect_block in self.bottleneck_block_list:
            y = bottlenect_block(y)

        y = self.pool2d_avg(y)
        y =paddle.reshape(y, [y.shape[0], -1])

        y = self.out(y)

        if label is not None:
            acc = paddle.metric.accuracy(y, label)
            return y, acc
        else:
            return y


# 定义训练过程
def train_pm(model, optimizer):
    # 开启0号GPU训练
    use_gpu = False
    paddle.set_device('gpu:0') if use_gpu else paddle.set_device('cpu')
    print("start training...")
    model.train()

    epoch_num = 10
    # 定义数据读取器，训练数据读取器和验证数据读取器
    train_loader = data_loader(DATADIR, batchsize=10, mode='train')
    valid_loader = valid_data_loader(DATADIR2, CSVFILE)
    iter = 0
    iters = []
    for epoch in range(epoch_num):
        model.train()
        for batch_id, data in enumerate(train_loader()):
            x_data, y_data = data
            img = paddle.to_tensor(x_data)
            label = paddle.to_tensor(y_data)

            # 前向计算
            logits, acc = model(img, label)
            loss = F.cross_entropy(logits, label)

            avg_loss = paddle.mean(loss)
            if batch_id % 10 == 0:
                # 使用visual DL进行绘图
                iters.append(iter)
                log_writer.add_scalar(tag='acc', step=iter, value=acc.numpy())
                log_writer.add_scalar(tag='loss', step=iter, value=avg_loss.numpy())
                print('epoch:{}, batch_id:{}, loss is:{}'.format(epoch, batch_id, avg_loss.numpy()))
            iter += 1

            # 反向传播，更新权重，清除梯度
            avg_loss.backward()
            optimizer.step()
            optimizer.clear_grad()


        accuracies = []
        losses = []
        for batch_id, data in enumerate(valid_loader()):
            x_data, y_data = data
            img = paddle.to_tensor(x_data)
            label = paddle.to_tensor(y_data)
            logits, acc = model(img, label)


            loss = F.cross_entropy(input=logits, label=label)
            avg_val_loss = paddle.mean(loss)
            accuracies.append(float(acc.numpy()))
            losses.append(float(avg_val_loss.numpy()))

        # 计算多个batch的平均损失和准确率
        acc_val_mean = np.array(accuracies).mean()
        avg_loss_val_mean = np.array(losses).mean()

        log_writer.add_scalar(tag='eval_acc', step=iter, value=acc_val_mean)

        print("loss={}, acc={}".format(avg_loss_val_mean, acc_val_mean))


    paddle.save(model.state_dict(), './work/GoogLeNet.pdparams')


# 创建模型
model = ResNet(class_dim=2)
# 启动训练过程
# opt = paddle.optimizer.Momentum(learning_rate=0.001, momentum=0.9, parameters=model.parameters())
opt = paddle.optimizer.Adam(learning_rate=0.00001, parameters=model.parameters())
# opt = paddle.optimizer.SGD(learning_rate=0.001, parameters=model.parameters())
train_pm(model, optimizer=opt)
