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

class VGG(paddle.nn.Layer):
    def __init__(self, num_classes=1):
        super(VGG, self).__init__()

        in_channels = [3, 64, 128, 256, 512, 512]
        # 定义第一个卷积块，包含两个卷积
        self.conv1_1 = Conv2D(in_channels=in_channels[0], out_channels=in_channels[1], kernel_size=3, padding=1, stride=1)
        self.conv1_2 = Conv2D(in_channels=in_channels[1], out_channels=in_channels[1], kernel_size=3, padding=1, stride=1)
        # 定义第二个卷积块，包含两个卷积
        self.conv2_1 = Conv2D(in_channels=in_channels[1], out_channels=in_channels[2], kernel_size=3, padding=1, stride=1)
        self.conv2_2 = Conv2D(in_channels=in_channels[2], out_channels=in_channels[2], kernel_size=3, padding=1, stride=1)
        # 定义第三个卷积块，包含三个卷积
        self.conv3_1 = Conv2D(in_channels=in_channels[2], out_channels=in_channels[3], kernel_size=3, padding=1, stride=1)
        self.conv3_2 = Conv2D(in_channels=in_channels[3], out_channels=in_channels[3], kernel_size=3, padding=1, stride=1)
        self.conv3_3 = Conv2D(in_channels=in_channels[3], out_channels=in_channels[3], kernel_size=3, padding=1, stride=1)
        # 定义第四个卷积块，包含三个卷积
        self.conv4_1 = Conv2D(in_channels=in_channels[3], out_channels=in_channels[4], kernel_size=3, padding=1, stride=1)
        self.conv4_2 = Conv2D(in_channels=in_channels[4], out_channels=in_channels[4], kernel_size=3, padding=1, stride=1)
        self.conv4_3 = Conv2D(in_channels=in_channels[4], out_channels=in_channels[4], kernel_size=3, padding=1, stride=1)
        # 定义第五个卷积块，包含三个卷积
        self.conv5_1 = Conv2D(in_channels=in_channels[4], out_channels=in_channels[5], kernel_size=3, padding=1, stride=1)
        self.conv5_2 = Conv2D(in_channels=in_channels[5], out_channels=in_channels[5], kernel_size=3, padding=1, stride=1)
        self.conv5_3 = Conv2D(in_channels=in_channels[5], out_channels=in_channels[5], kernel_size=3, padding=1, stride=1)

        # 使用Sequential 将全连接层和relu组成线性结构(fc+relu)
        # 当输入为224*224时，经过5个卷积块和池化层后，形状变为512*7*7
        self.fc1 = paddle.nn.Sequential(
            paddle.nn.Linear(512*7*7, 4096),
            paddle.nn.ReLU()
        )
        self.drop1_ratio = 0.5
        self.dropout1 = paddle.nn.Dropout(self.drop1_ratio, mode='upscale_in_train')
        # 使用Sequential将全连接层和relu组成一个线性结构（fc+relu）
        self.fc2 = paddle.nn.Sequential(
            paddle.nn.Linear(4096, 4096),
            paddle.nn.ReLU()
        )
        self.drop2_ratio = 0.5
        self.dropout2 = paddle.nn.Dropout(self.drop2_ratio, mode='upscale_in_train')
        self.fc3 = paddle.nn.Linear(4096, num_classes)

        self.relu = paddle.nn.ReLU()
        self.pool = MaxPool2D(stride=2, kernel_size=2)

        self.softmax = Softmax()



    def forward(self, x, label=None):
        x = self.relu(self.conv1_1(x))
        x = self.relu(self.conv1_2(x))
        x = self.pool(x)

        x = self.relu(self.conv2_1(x))
        x = self.relu(self.conv2_2(x))
        x = self.pool(x)

        x = self.relu(self.conv3_1(x))
        x = self.relu(self.conv3_2(x))
        x = self.relu(self.conv3_3(x))
        x = self.pool(x)

        x = self.relu(self.conv4_1(x))
        x = self.relu(self.conv4_2(x))
        x = self.relu(self.conv4_3(x))
        x = self.pool(x)

        x = self.relu(self.conv5_1(x))
        x = self.relu(self.conv5_2(x))
        x = self.relu(self.conv5_3(x))
        x = self.pool(x)

        x = paddle.flatten(x, 1, -1)
        x = self.dropout1(self.relu(self.fc1(x)))
        x = self.dropout2(self.relu(self.fc2(x)))
        x = self.fc3(x)

        # x = self.softmax(x)

        if label is not None:
            # print(x)
            # print(label)
            acc = paddle.metric.accuracy(input=x, label=label)
            return x, acc
        else:
            return x


# 定义训练过程
def train_pm(model, optimizer):
    # 开启0号GPU训练
    use_gpu = True
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


    paddle.save(model.state_dict(), './work/VGG.pdparams')


# 创建模型
model = VGG(num_classes=2)
# 启动训练过程
# opt = paddle.optimizer.Momentum(learning_rate=0.001, momentum=0.9, parameters=model.parameters())
# opt = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())
opt = paddle.optimizer.SGD(learning_rate=0.001, parameters=model.parameters())
train_pm(model, optimizer=opt)
