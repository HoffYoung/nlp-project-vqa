import mindspore.nn as nn
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore.ops import operations as P

class VGG(nn.Cell):
    """
    VGG网络结构
    """
    def __init__(self, num_class = 1024):
        super(VGG, self).__init__()
        self.in_channels = 3
        self.input_perm = (0, 3, 1, 2)
        
        self.trans = P.Transpose()
        self.conv1 = nn.Conv2d(self.in_channels, 64, 3, pad_mode='same')
        self.conv2 = nn.Conv2d(64, 64, 3, pad_mode='same')
        self.conv3 = nn.Conv2d(64, 128, 3, pad_mode='same')
        self.conv4 = nn.Conv2d(128, 128, 3, pad_mode='same')
        self.conv5 = nn.Conv2d(128, 256, 3, pad_mode='same')
        self.conv6 = nn.Conv2d(256, 256, 3, pad_mode='same')
        self.conv7 = nn.Conv2d(256, 512, 3, pad_mode='same')
        self.conv8 = nn.Conv2d(512, 512, 3, pad_mode='same')
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Dense(7*7*512, 4096)
        self.fc2 = nn.Dense(4096, 4096)
        self.fc3 = nn.Dense(4096, num_class)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        
    def construct(self, x):
        # 使用定义好的运算构建前向网络
        x = self.trans(x, self.input_perm)
        # (224, 224, 3) -> (224, 224, 64)
        x = self.conv1(x)
        # (224, 224, 64) -> (224, 224, 64)
        x = self.conv2(x)
        # (224, 224, 64) -> (112, 112, 64)
        x = self.max_pool2d(x)
        # (112, 112, 64) -> (112, 112, 128)
        x = self.conv3(x)
        # (112, 112, 128) -> (112, 112, 128)
        x = self.conv4(x)
        # (112, 112, 128) -> (56, 56, 128)
        x = self.max_pool2d(x)
        # (56, 56, 128) -> (56, 56, 256)
        x = self.conv5(x)
        # (56, 56, 256) -> (56, 56, 256)
        x = self.conv6(x)
        # (56, 56, 256) -> (28, 28, 256)
        x = self.conv6(x)
        # (56, 56, 256) -> (28, 28, 256)
        x = self.max_pool2d(x)
        # (28, 28, 256) -> (28, 28, 512)
        x = self.conv7(x)
        # (28, 28, 512) -> (28, 28, 512)
        x = self.conv8(x)
        # (28, 28, 512) -> (28, 28, 512)
        x = self.conv8(x)
        # (28, 28, 512) -> (14, 14, 512)
        x = self.max_pool2d(x)
        # (14, 14, 512) -> (14, 14, 512)
        x = self.conv8(x)
        # (14, 14, 512) -> (14, 14, 512)
        x = self.conv8(x)
        # (14, 14, 512) -> (14, 14, 512)
        x = self.conv8(x)
        # (14, 14, 512) -> (7, 7, 512)
        x = self.max_pool2d(x)
        # (7, 7, 512) -> (7*7*512)
        x = self.flatten(x)
        # fully connected + ReLU
        # (7*7*512) -> (1, 1, 4096)
        x = self.fc1(x)
        x = self.relu(x)
        # (1, 1, 4096) -> (1, 1, 4096)
        x = self.fc2(x)
        x = self.relu(x)
        # (1, 1, 4096) -> (1, 1, 1024)
        x = self.fc3(x)
        x = self.relu(x) 

        return x

class LSTM(nn.Cell):
    """
    LSTM结构
    """
    def __init__(self, input_size = 25, batch_size = 32, hidden_size = 512, num_layers = 2, dropout = 0.1, bidirectional = False):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, input_size)
        self.net = nn.LSTM(input_size, hidden_size, num_layers, has_bias = True, batch_first = True, dropout = dropout, bidirectional = bidirectional)
        self.h0 = Tensor(np.ones([(2 if bidirectional else 1) * num_layers, batch_size, hidden_size]).astype(np.float32))
        self.c0 = Tensor(np.ones([(2 if bidirectional else 1) * num_layers, batch_size, hidden_size]).astype(np.float32))
        self.concat = P.Concat()
        self.trans = P.Transpose()
        self.flatten = nn.Flatten()
        self.fc = nn.Dense(2*2*512, 1024)
        
    def construct(self, x):
        x = self.embedding(x)
        output, (hn, cn) = self.net(x, (self.h0, self.c0))
        x = self.concat((hn, cn))
        x = self.trans(x, (1, 0, 2))
        x = self.flatten(x)
        x = self.fc(x)
        return x



class VQABasic(nn.Cell):
    """
    VQABsic网络结构
    """
    def __init__(self):
        super(VQABasic, self).__init__()
        self.VGGnet = VGG()
        self.LSTMnet = LSTM()
        self.mul = P.Mul()
        self.fc = nn.Dense(1024, 1000)
        self.softmax = nn.Softmax()
        
    def construct(self, x_image, x_question):
        x_image = self.VGGnet(x_image)
        x_question = self.LSTMnet(x_question)
        # Point-wise multiplication
        output = self.mul(x_image, x_question)
        # Fully connected
        output = self.fc(output)
        return self.softmax(output)


if __name__ == "__main__":
	VQABasicNet = VQABasic()
	aout = VQABasicNet.construct(Tensor(train_images_list[0:32], dtype=mstype.float32), Tensor(train_questions_vec[0:32], dtype=mstype.int32))
