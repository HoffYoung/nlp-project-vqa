import numpy as np
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import Parameter
from mindspore import load_checkpoint
from mindspore import dtype as mstype
from mindspore.ops import operations as P
from pretrained.VGG import VGG

class LSTM(nn.Cell):
    """
    LSTM结构
    """
    def __init__(self, config, hidden_size = 512, num_layers = 2, dropout = 0.1, bidirectional = False):
        super(LSTM, self).__init__()
        self.net = nn.LSTM(1, hidden_size, num_layers, has_bias = True, batch_first = True, dropout = dropout, bidirectional = bidirectional)
        self.h0 = Tensor(np.zeros([(2 if bidirectional else 1) * num_layers, config.batch_size, hidden_size]).astype(np.float32))
        self.c0 = Tensor(np.zeros([(2 if bidirectional else 1) * num_layers, config.batch_size, hidden_size]).astype(np.float32))
        self.concat = P.Concat(axis=0)
        self.expand = P.ExpandDims()
        self.trans = P.Transpose()
        self.flatten = nn.Flatten()
        self.fc = nn.Dense(2*2*512, 1024, activation='tanh')
        
    def construct(self, x):
        # x: (batch_size, seq_length) -> (batch_size, seq_length, 1)
        x = self.expand(x, -1)
        # hn, cn: (num_directions * num_layers, batch_size, hidden_size)
        output, (hn, cn) = self.net(x, (self.h0, self.c0))
        # x: (4, batch_size, hidden_size)
        x = self.concat((hn, cn))
        # x: (batch_size, 4, hidden_size)
        x = self.trans(x, (1, 0, 2))
        # x: (batch_size, 4 * hidden_size)
        x = self.flatten(x)
        # x: (batch_size, 1024)
        x = self.fc(x)
        return x

class VQABasic(nn.Cell):
    """
    VQABsic网络结构
    """
    def __init__(self, config):
        super(VQABasic, self).__init__()
        self.VGGnet = VGG(config)
        self.LSTMnet = LSTM(config)
        self.mul = P.Mul()
        self.fc1 = nn.Dense(1024, config.output_size, activation='tanh')
        self.fc2 = nn.Dense(config.output_size, config.output_size, activation='tanh')
        self.fc3 = nn.Dense(4096, 1024, activation='tanh')
        self.l2Normalize = P.L2Normalize()
        self.dropout = nn.Dropout(0.5)
        self.cast = P.Cast()
        
    def construct(self, x_image, x_question):
        x_image = self.VGGnet(x_image)
        # l2 normalization for image
        x_image = self.l2Normalize(x_image)
        # (4096,) -> (1024,)
        x_image = self.fc3(x_image) # tanh
        
        x_question = self.cast(x_question, mstype.float32)
        x_question = self.LSTMnet(x_question)
        # Point-wise multiplication
        output = self.mul(x_image, x_question)
        # Fully connected
        output = self.fc1(output)
        output = self.fc2(output)
        output = self.dropout(output)

        return output

class VQABasicOpAttn(nn.Cell):
    """
    VQABsic with Options Attention 网络结构
    """
    def __init__(self, config):
        super(VQABasicOpAttn, self).__init__()
        self.VGGnet = VGG(config)
        self.LSTMnet = LSTM(config)
        self.embeddings = nn.Embedding(config.vocab_size, 1024)
        self.mul = P.Mul()
        self.fc1 = nn.Dense(1024, config.output_size, activation='tanh')
        self.fc2 = nn.Dense(config.output_size, config.output_size, activation='tanh')
        self.fc3 = nn.Dense(4096, 1024, activation='tanh')
        self.l2Normalize = P.L2Normalize()
        self.dropout = nn.Dropout(0.5)
        self.cast = P.Cast()
        self.fc_query = nn.Dense(1024, 1024, has_bias=False)
        self.fc_key = nn.Dense(1024, 1024, has_bias=True)
        self.tanh = nn.Tanh()
        self.fc_h = nn.Dense(1024, 1, has_bias=True)
        self.softmax = nn.Softmax(axis=1)
        self.matmul = P.BatchMatMul(transpose_a=True)
        self.reshape = P.Reshape()
        self.expand_dims = P.ExpandDims()
        
    def construct(self, x_image, x_question, x_options):
        x_image = self.VGGnet(x_image)
        # l2 normalization for image
        x_image = self.l2Normalize(x_image)
        # (4096,) -> (1024,)
        x_image = self.fc3(x_image) # tanh
        
        x_question = self.cast(x_question, mstype.float32)
        x_question = self.LSTMnet(x_question)
        # Point-wise multiplication
        output = self.mul(x_image, x_question)
        
        # Options (batch_size, 10, 1024)
        options = self.embeddings(x_options)
        output = self.expand_dims(output, 1)
        # (batch_size, 1, 1024) + (batch_size, 10, 1024) -> (batch_size, 10, 1024)
        h = self.tanh(self.fc_query(output) + self.fc_key(options))
        # (batch_size, 10, 1)
        p = self.softmax(self.fc_h(h))
        # (batch_size, 10, 1).T @ (batch_size, 10, 1024) -> (batch_size, 1, 1024)
        new_context_vec = self.matmul(p, options)
        # (batch_size, 1, 1024) -> (batch_size, 1024)
        new_context_vec = self.reshape(new_context_vec, (-1, 1024))
 
        # Fully connected
        new_context_vec = self.fc1(new_context_vec)
        new_context_vec = self.fc2(new_context_vec)
        new_context_vec = self.dropout(new_context_vec)

        return new_context_vec

if __name__ == "__main__":
	VQABasicNet = VQABasic()
	aout = VQABasicNet.construct(Tensor(train_images_list[0:32], dtype=mstype.float32), Tensor(train_questions_vec[0:32], dtype=mstype.int32))
