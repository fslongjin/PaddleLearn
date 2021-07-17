import paddle.vision
from paddle.vision.models import resnet50
from paddle.vision.datasets import Cifar10
from paddle.optimizer import Adam
from paddle.regularizer import L2Decay
from paddle.nn import CrossEntropyLoss
from paddle.metric import Accuracy
from paddle.vision.transforms import Transpose

# 确保从paddle.vision.datasets中读取的图像数据是numpy.ndarray类型
paddle.vision.set_image_backend('cv2')

# 调用resnet50
model = paddle.Model(resnet50(pretrained=False, num_classes=10))

# 使用cifar-10数据集
train_dataset = Cifar10(mode='train', transform=Transpose())
val_dataset = Cifar10(mode='test', transform=Transpose())

opt = Adam(learning_rate=0.0001, weight_decay=L2Decay(1e-4), parameters=model.parameters())

# 进行训练前准备
model.prepare(opt, CrossEntropyLoss(), Accuracy(topk=(1, 5)))
# 启动训练
model.fit(train_dataset, val_dataset, epochs=50, batch_size=64, save_dir='work/', num_workers=8)
