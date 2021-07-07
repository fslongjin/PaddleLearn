import paddle
from paddle.nn import Linear
import paddle.nn.functional as F
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
# 确保从paddle.vision.datasets.MNIST中加载的图像数据是np.ndarray类型
paddle.vision.set_image_backend('cv2')


class MNIST(paddle.nn.Layer):
    def __init__(self):
        super(MNIST, self).__init__()

        self.fc = Linear(in_features=784, out_features=1)

    def forward(self, inputs):
        opt = self.fc(inputs)
        return opt


def train(model):

    model.train()
    train_loader = paddle.io.DataLoader(paddle.vision.datasets.MNIST(mode='train'),
                                        batch_size=16,
                                        shuffle=True)

    opt = paddle.optimizer.SGD(learning_rate=0.001, parameters=model.parameters())

    EPOCH_NUM = 5
    for epoch in range(EPOCH_NUM):
        for batch_id, data in enumerate(train_loader()):
            images = norm_img(data[0].astype('float32'))
            labels = data[1].astype('float32')

            predicts = model.forward(images)

            loss = F.square_error_cost(predicts, labels)

            avg_loss = paddle.mean(loss)

            if batch_id % 20 == 0:
                print("epoch_id:{}, batch_id:{}, loss:{}".format(epoch, batch_id, avg_loss.numpy()))

            # 反向传播
            avg_loss.backward()
            # 最小化loss，更新参数
            opt.step()
            # 清除梯度
            opt.clear_grad()





# 图像归一化
def norm_img(img):
    assert len(img.shape) == 3
    batch_size, img_h, img_w = img.shape[0], img.shape[1], img.shape[2]

    # 归一化图像数据
    img = img/127.5 - 1
    img = paddle.reshape(img, [batch_size, img_h*img_w])

    return img


def t_test(img_path='example_0.png'):


    im = Image.open(img_path).convert('L')


    print("原始图像shape：", np.array(im).shape)

    # 使用Image.ANTIALIAS方式采样原始图片
    im = im.resize((28, 28), Image.ANTIALIAS)
    print("采样后图片shape：", np.array(im).shape)
    im = np.array(im).reshape(1, -1).astype('float32')
    print("reshape后图片shape：", np.array(im).shape)
    # 图像归一化
    im = im/127.5 - 1

    param_dict = paddle.load('mnist_linear.pdparams')

    model.load_dict(param_dict)

    model.eval()
    result = model(paddle.to_tensor(im))

    # 输出预测结果
    print("本次预测的数字是：", result.numpy().astype('int32'))



if __name__ == '__main__':
    mode = 'train'
    if mode == 'train':
        model = MNIST()
        train(model)
        paddle.save(model.state_dict(), 'mnist_linear.pdparams')
        print("模型保存成功！")
    elif mode == 'test':
        model = MNIST()
        t_test()

