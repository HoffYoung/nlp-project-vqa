import numpy as np
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import Parameter
from mindspore import load_checkpoint
from mindspore import dtype as mstype
from mindspore.ops import operations as P

class VGG(nn.Cell):
    """
    VGG网络结构
    """
    def __init__(self, config):
        super(VGG, self).__init__()
        self.in_channels = 3
        self.input_perm = (0, 3, 1, 2)
        self.model_name = config.model

        # load pretrained ckpt
        params = load_checkpoint(config.pretrained_path)
        params["layers.0.weight"].requires_grad = False
        params["layers.2.weight"].requires_grad = False
        params["layers.5.weight"].requires_grad = False
        params["layers.7.weight"].requires_grad = False
        params["layers.10.weight"].requires_grad = False
        params["layers.12.weight"].requires_grad = False
        params["layers.14.weight"].requires_grad = False
        params["layers.16.weight"].requires_grad = False
        params["layers.19.weight"].requires_grad = False
        params["layers.21.weight"].requires_grad = False
        params["layers.23.weight"].requires_grad = False
        params["layers.25.weight"].requires_grad = False
        params["layers.28.weight"].requires_grad = False
        params["layers.30.weight"].requires_grad = False
        params["layers.32.weight"].requires_grad = False
        params["layers.34.weight"].requires_grad = False
        params["classifier.0.weight"].requires_grad = False
        params["classifier.0.weight"].requires_grad = False
        params["classifier.3.bias"].requires_grad = False
        params["classifier.3.bias"].requires_grad = False

        self.cast = P.Cast()
        self.trans = P.Transpose()
        self.conv1 = nn.Conv2d(self.in_channels, 64, 3, pad_mode='same', weight_init=params["layers.0.weight"])
        self.conv2 = nn.Conv2d(64, 64, 3, pad_mode='same', weight_init=params["layers.2.weight"])
        self.conv3 = nn.Conv2d(64, 128, 3, pad_mode='same', weight_init=params["layers.5.weight"])
        self.conv4 = nn.Conv2d(128, 128, 3, pad_mode='same', weight_init=params["layers.7.weight"])
        self.conv5 = nn.Conv2d(128, 256, 3, pad_mode='same', weight_init=params["layers.10.weight"])
        self.conv6 = nn.Conv2d(256, 256, 3, pad_mode='same', weight_init=params["layers.12.weight"])
        self.conv7 = nn.Conv2d(256, 256, 3, pad_mode='same', weight_init=params["layers.14.weight"])
        self.conv8 = nn.Conv2d(256, 256, 3, pad_mode='same', weight_init=params["layers.16.weight"])
        self.conv9 = nn.Conv2d(256, 512, 3, pad_mode='same', weight_init=params["layers.19.weight"])
        self.conv10 = nn.Conv2d(512, 512, 3, pad_mode='same', weight_init=params["layers.21.weight"])
        self.conv11 = nn.Conv2d(512, 512, 3, pad_mode='same', weight_init=params["layers.23.weight"])
        self.conv12 = nn.Conv2d(512, 512, 3, pad_mode='same', weight_init=params["layers.25.weight"])
        self.conv13 = nn.Conv2d(512, 512, 3, pad_mode='same', weight_init=params["layers.28.weight"])
        self.conv14 = nn.Conv2d(512, 512, 3, pad_mode='same', weight_init=params["layers.30.weight"])
        self.conv15 = nn.Conv2d(512, 512, 3, pad_mode='same', weight_init=params["layers.32.weight"])
        self.conv16 = nn.Conv2d(512, 512, 3, pad_mode='same', weight_init=params["layers.34.weight"])
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Dense(7*7*512, 4096, weight_init=params["classifier.0.weight"], bias_init=params["classifier.0.bias"])
        self.fc2 = nn.Dense(4096, 4096, weight_init=params["classifier.3.weight"], bias_init=params["classifier.3.bias"])
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        
    def construct(self, x):
        x = self.cast(x, mstype.float32)
        # 使用定义好的运算构建前向网络
        x = self.trans(x, self.input_perm)
        # (224, 224, 3) -> (224, 224, 64)
        x = self.relu(self.conv1(x))
        # (224, 224, 64) -> (224, 224, 64)
        x = self.relu(self.conv2(x))
        # (224, 224, 64) -> (112, 112, 64)
        x = self.max_pool2d(x)
        # (112, 112, 64) -> (112, 112, 128)
        x = self.relu(self.conv3(x))
        # (112, 112, 128) -> (112, 112, 128)
        x = self.relu(self.conv4(x))
        # (112, 112, 128) -> (56, 56, 128)
        x = self.max_pool2d(x)
        # (56, 56, 128) -> (56, 56, 256)
        x = self.relu(self.conv5(x))
        # (56, 56, 256) -> (56, 56, 256)
        x = self.relu(self.conv6(x))
        # (56, 56, 256) -> (56, 56, 256)
        x = self.relu(self.conv7(x))
        # (56, 56, 256) -> (56, 56, 256)
        x = self.relu(self.conv8(x))
        # (56, 56, 256) -> (28, 28, 256)
        x = self.max_pool2d(x)
        # (28, 28, 256) -> (28, 28, 512)
        x = self.relu(self.conv9(x))
        # (28, 28, 512) -> (28, 28, 512)
        x = self.relu(self.conv10(x))
        # (28, 28, 512) -> (28, 28, 512)
        x = self.relu(self.conv11(x))
        # (28, 28, 512) -> (28, 28, 512)
        x = self.relu(self.conv12(x))
        # (28, 28, 512) -> (14, 14, 512)
        x = self.max_pool2d(x)
        # (14, 14, 512) -> (14, 14, 512)
        x = self.relu(self.conv13(x))
        # (14, 14, 512) -> (14, 14, 512)
        x = self.relu(self.conv14(x))
        # (14, 14, 512) -> (14, 14, 512)
        x = self.relu(self.conv15(x))
        # (14, 14, 512) -> (14, 14, 512)
        x = self.relu(self.conv16(x))
        # (14, 14, 512) -> (7, 7, 512)
        x = self.max_pool2d(x)
        if self.model_name == 'stack_attention' or self.model_name == 'topdown_attention':
            # return (14, 14, 512) (input_size (448,448,3))
            return x
        # (7, 7, 512) -> (7*7*512)
        x = self.flatten(x)
        # fully connected + ReLU
        # (7*7*512) -> (4096,)
        x = self.fc1(x)
        x = self.relu(x)
        # (4096,) -> (4096,)
        x = self.fc2(x)
        x = self.relu(x)

        return x