import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import paddle
from paddle.nn import Conv2D
from paddle.nn.initializer import Assign

img = Image.open('work/000000098520.jpg')

# 设置卷积核参数
w = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32')
w = w.reshape([1, 1, 3, 3])
# 由于输入通道是3， 将卷积核的形状从[1,1,3,3]调整为[1,3,3,3]
w = np.repeat(w, 3, axis=1)
# 创建卷积算子
conv = Conv2D(in_channels=3, out_channels=1, kernel_size=[3, 3],
              weight_attr=paddle.ParamAttr(initializer=Assign(value=w)))

# 将读入的图片转化为float32类型的numpy.ndarray
x = np.array(img).astype('float32')
# 图片读入成ndarray时，形状是[H,W,3]
# 将通道这一维度调整到最前面
x = np.transpose(x, (2, 0, 1))
# 数据形状调整为[N,C,H,W]形式
x = x.reshape(1, 3, img.height, img.width)
x = paddle.to_tensor(x)

y = conv(x)
out = y.numpy()

plt.figure(figsize=(20, 10))
f = plt.subplot(121)
f.set_title('input image', fontsize=15)
plt.imshow(img)

f = plt.subplot(122)
f.set_title('output feature image', fontsize=15)
plt.imshow(out.squeeze(), cmap='gray')
plt.show()

