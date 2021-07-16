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


# 定义 Inception块
class Inception(paddle.nn.Layer):
    def __init__(self, c0, c1, c2, c3, c4, **kwargs):
        """
        Inception模块的实现代码，

        c1,图(b)中第一条支路1x1卷积的输出通道数，数据类型是整数
        c2,图(b)中第二条支路卷积的输出通道数，数据类型是tuple或list,
               其中c2[0]是1x1卷积的输出通道数，c2[1]是3x3
        c3,图(b)中第三条支路卷积的输出通道数，数据类型是tuple或list,
               其中c3[0]是1x1卷积的输出通道数，c3[1]是3x3
        c4,图(b)中第一条支路1x1卷积的输出通道数，数据类型是整数
        """
        super(Inception, self).__init__()
        # 依次创建Inception块每条支路上使用到的操作
        self.p1_1 = Conv2D(in_channels=c0, out_channels=c1, kernel_size=1, stride=1)
        self.p2_1 = Conv2D(in_channels=c0, out_channels=c2[0], kernel_size=1, stride=1)
        self.p2_2 = Conv2D(in_channels=c2[0], out_channels=c2[1], kernel_size=3, padding=1, stride=1)
        self.p3_1 = Conv2D(in_channels=c0, out_channels=c3[0], kernel_size=1, stride=1)
        self.p3_2 = Conv2D(in_channels=c3[0], out_channels=c3[1], kernel_size=5, padding=2, stride=1)
        self.p4_1 = MaxPool2D(kernel_size=3, stride=1, padding=1)
        self.p4_2 = Conv2D(in_channels=c0, out_channels=c4, kernel_size=1, stride=1)

        # 新加一层Batch norm稳定收敛
        # self.batchnorm = paddle.nn.BatchNorm2D(c1+c2[1]+c3[1]+c4)

    def forward(self, x, label=None):
        # 支路1只包含一个1*1卷积
        p1 = F.relu(self.p1_1(x))
        # 支路2包含 1*1卷积+3*3卷积
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        # 支路3包含 1*1卷积+5*5卷积
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        # 支路4包含最大池化和1*1卷积
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        # 将每个支路的输出特征图拼接在一起作为最终的输出结果
        return paddle.concat([p1, p2, p3, p4], axis=1)

class GoogLeNet(paddle.nn.Layer):
    def __init__(self, num_classes=1):
        super(GoogLeNet, self).__init__()

        # GoogLeNet包含5个模块，每个模块后面紧跟一个池化层
        # 第一个模块包含一个卷积层
        self.conv1 = Conv2D(in_channels=3, out_channels=64, kernel_size=7, padding=3, stride=1)
        # 3*3最大池化
        self.pool1 = MaxPool2D(kernel_size=3, stride=2, padding=1)

        # 第二个模块包含2个卷积层
        self.conv2_1 = Conv2D(in_channels=64, out_channels=64, kernel_size=1, stride=1)
        self.conv2_2 = Conv2D(in_channels=64, out_channels=192, kernel_size=3, padding=1, stride=1)
        # 3*3最大池化
        self.pool2 = MaxPool2D(kernel_size=3, stride=2, padding=1)
        # 第三个模块包含2个Inception模块
        self.block3_1 = Inception(192, 64, (96, 128), (16, 32), 32)
        self.block3_2 = Inception(256, 128, (128, 192), (32, 96), 64)
        # 3*3最大池化
        self.pool3 = MaxPool2D(kernel_size=3, stride=2, padding=1)

        # 第四个模块包含5个Inception块
        self.block4_1 = Inception(480, 192, (96, 208), (16, 48), 64)
        self.block4_2 = Inception(512, 160, (112, 224), (24, 64), 64)
        self.block4_3 = Inception(512, 128, (128, 256), (24, 64), 64)
        self.block4_4 = Inception(512, 112, (144, 288), (32, 64), 64)
        self.block4_5 = Inception(528, 256, (160, 320), (32, 128), 128)
        # 3*3最大池化
        self.pool4 = MaxPool2D(kernel_size=3, stride=2, padding=1)
        # 第五个模块包含2个Inception块
        self.block5_1 = Inception(832, 256, (160, 320), (32, 128), 128)
        self.block5_2 = Inception(832, 384, (192, 384), (48, 128), 128)
        # 全局池化，使用的是global_pooling， 不需要设置pool_stride
        self.pool5 = paddle.nn.AdaptiveAvgPool2D(output_size=1)
        self.fc = Linear(in_features=1024, out_features=num_classes)


    def forward(self, x, label=None):
        x = self.pool1(F.relu(self.conv1(x)))

        x = self.pool2(F.relu(self.conv2_2(F.relu(self.conv2_1(x)))))

        x = self.pool3(self.block3_2(self.block3_1(x)))

        x = self.block4_3(self.block4_2(self.block4_1(x)))
        x = self.pool4(self.block4_5(self.block4_4(x)))

        x = self.pool5(self.block5_2(self.block5_1(x)))

        x = paddle.reshape(x, [x.shape[0], -1])
        x = self.fc(x)

        if label is not None:
            acc = paddle.metric.accuracy(input=x, label=label)
            return x, acc
        else:
            return x



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
model = GoogLeNet(num_classes=2)
# 启动训练过程
# opt = paddle.optimizer.Momentum(learning_rate=0.001, momentum=0.9, parameters=model.parameters())
# opt = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())
opt = paddle.optimizer.SGD(learning_rate=0.001, parameters=model.parameters())
train_pm(model, optimizer=opt)
