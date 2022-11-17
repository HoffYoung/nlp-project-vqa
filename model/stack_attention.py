import numpy as np
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import Parameter
from mindspore import load_checkpoint
from mindspore import dtype as mstype
from mindspore.ops import operations as P
from pretrained.VGG import VGG
from pretrained.embeddings import *

def _weight_variable(shape, factor=0.01):
    init_value = np.random.randn(*shape).astype(np.float32) * factor
    return Tensor(init_value)

class CNN(nn.Cell):
	"""
	TextCNN for SAN
	"""

	def __init__(self, vocab_dict, config):
		super(CNN, self).__init__()
		embedding_table = Tensor(create_embedding_table(vocab_dict, config))
		self.embeddings = nn.Embedding(config.vocab_size, 200, embedding_table=embedding_table)
		self.layer1 = self.make_layer(out_channels=256, kernel_height=1, max_length=config.max_length)
		self.layer2 = self.make_layer(out_channels=512, kernel_height=2, max_length=config.max_length)
		self.layer3 = self.make_layer(out_channels=512, kernel_height=3, max_length=config.max_length)
		self.reducemean = P.ReduceMax(keep_dims=False)
		self.expand_dims = P.ExpandDims()
		self.concat = P.Concat(axis=1)
		self.drop = nn.Dropout(0.5)
		self.fc1 = nn.Dense(240, 640)
		self.fc2 = nn.Dense(640, 1280)

	def make_layer(self, out_channels, kernel_height, max_length):
		return nn.SequentialCell(
            [
                nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=(kernel_height, 200), 
				          weight_init=_weight_variable((out_channels, 1, kernel_height, 200)), padding=1, pad_mode="pad"),
                nn.Tanh(),
                nn.MaxPool2d(kernel_size=(max_length-kernel_height+1,1)),
            ]
        )

	def construct(self, questions):
		# x: (batch_size, 1, max_length)
		x = self.expand_dims(questions, 1)
		# x: (batch_size, 1, max_length, hidden_size)
		x = self.embeddings(x)
		x1 = self.layer1(x) # unigram
		x2 = self.layer2(x) # bigram
		x3 = self.layer3(x) # trigram
		
		x1 = self.reducemean(x1, (2, 3)) # (batch_size, 256)
		x2 = self.reducemean(x2, (2, 3)) # (batch_size, 512)
		x3 = self.reducemean(x3, (2, 3)) # (batch_size, 512)
		
		# x: (batch_size, 240)
		x = self.concat((x1, x2, x3))
		# x: (batch_size, 640)
		#x = self.fc1(x)
		# x: (batch_size, 1280)
		#x = self.fc2(x)
		x = self.drop(x)
		
		return x


class StackedAttentionLayer(nn.Cell):
	"""
	Single layer of stacked attention
	"""
	def __init__(self):
		super(StackedAttentionLayer, self).__init__()
		self.fc_image = nn.Dense(1280, 1280, has_bias=False)
		self.fc_comb = nn.Dense(1280, 1280, has_bias=True)
		self.tanh = nn.Tanh()
		self.fc_h = nn.Dense(1280, 1, has_bias=True)
		self.softmax = nn.Softmax(axis=1)
		self.matmul = P.BatchMatMul(transpose_a=True)
	
	def construct(self, images_vec, combined_vec):
		# (batch_size, 196, 1280) + (batch_size, 1, 1280) -> (batch_size, 196, 1280)
		h = self.tanh(self.fc_image(images_vec) + self.fc_comb(combined_vec))
		# (batch_size, 196, 1)
		p = self.softmax(self.fc_h(h))
		# (batch_size, 196, 1).T @ (batch_size, 196, 1280) -> (batch_size, 1, 1280)
		new_images_vec = self.matmul(p, images_vec)
		# (batch_size, 1, 1280) + (batch_size, 1, 1280) -> (batch_size, 1, 1280)
		new_combined_vec = new_images_vec + combined_vec
		
		return new_combined_vec


class StackedAttentionNet(nn.Cell):
	"""
	Stacked Attention Network
	Based on the paper: Stacked Attention Networks for Image Question Answering
	"""
	def __init__(self, vocab_dict, config):
		super(StackedAttentionNet, self).__init__()
		self.VGGnet = VGG(config)
		self.text_net = CNN(vocab_dict, config)
		self.san_layer1 = StackedAttentionLayer()
		self.san_layer2 = StackedAttentionLayer()
		self.fc_i = nn.Dense(in_channels=512, out_channels=1280, has_bias=True)
		self.fc_out = nn.Dense(in_channels=1280, out_channels=config.output_size, has_bias=True)
		self.reshape = P.Reshape()
		self.expand_dims = P.ExpandDims()

	def construct(self, images, questions):
		# images: (batch_size, 14, 14, 512)
		images = self.VGGnet(images)
		# images: (batch_size, 196, 512)
		images = self.reshape(images, (-1, 196, 512))
		# images: (batch_size, 196, 1280)
		images = self.fc_i(images)
		# questions: (batch_size, 1280)
		questions = self.text_net(questions)
		# questions: (batch_size, 1, 1280)
		questions = self.expand_dims(questions, 1)
		# san layer 1
		combined_vec = self.san_layer1(images, questions)
		# san layer 2
		combined_vec = self.san_layer2(images, combined_vec)
		# flatten_vec: (batch_size, 1, 1280) -> (batch_size, 1280)
		flatten_vec = self.reshape(combined_vec, (-1, 1280))
		# output: (batch_size, 1280) -> (batch_size, output_size)
		preds = self.fc_out(flatten_vec)
		
		return preds

class StackedAttentionNetOpAttn(nn.Cell):
	"""
	Stacked Attention Network with Options Attention
	Based on the paper: Stacked Attention Networks for Imageg Question Answering
	"""
 
	def __init__(self, vocab_dict, config):
		super(StackedAttentionNetOpAttn, self).__init__()
		self.VGGnet = VGG(config)
		self.text_net = CNN(vocab_dict, config)
		self.embeddings = nn.Embedding(config.vocab_size, 1280)
		self.san_layer1 = StackedAttentionLayer()
		self.san_layer2 = StackedAttentionLayer()
		self.fc_i = nn.Dense(in_channels=512, out_channels=1280, has_bias=True)
		self.fc_out = nn.Dense(in_channels=1280, out_channels=config.output_size, has_bias=True)
		self.reshape = P.Reshape()
		self.expand_dims = P.ExpandDims()
		self.fc_query = nn.Dense(1280, 1280, has_bias=False)
		self.fc_key = nn.Dense(1280, 1280, has_bias=True)
		self.tanh = nn.Tanh()
		self.fc_h = nn.Dense(1280, 1, has_bias=True)
		self.softmax = nn.Softmax(axis=1)
		self.matmul = P.BatchMatMul(transpose_a=True)
		self.reshape = P.Reshape()

	def construct(self, images, questions, options):
		# images: (batch_size, 14, 14, 512)
		images = self.VGGnet(images)
		# images: (batch_size, 196, 512)
		images = self.reshape(images, (-1, 196, 512))
		# images: (batch_size, 196, 1280)
		images = self.fc_i(images)
		# questions: (batch_size, 1280)
		questions = self.text_net(questions)
		# questions: (batch_size, 1, 1280)
		questions = self.expand_dims(questions, 1)
		# san layer 1
		combined_vec = self.san_layer1(images, questions)
		# san layer 2
		# (batch_size, 1, 1280)
		combined_vec = self.san_layer2(images, combined_vec)
		# Options (batch_size, 10, 1280)
		options = self.embeddings(options)
		# (batch_size, 1, 1280) + (batch_size, 10, 1280) -> (batch_size, 10, 1280)
		h = self.tanh(self.fc_query(combined_vec) + self.fc_key(options))
		# (batch_size, 10, 1)
		p = self.softmax(self.fc_h(h))
		# (batch_size, 10, 1).T @ (batch_size, 10, 1280) -> (batch_size, 1, 1280) 
		new_context_vec = self.matmul(p, options)
		# (batch_size, 1, 1280) -> (batch_size, 1280)
		new_context_vec = self.reshape(new_context_vec, (-1, 1280))
		# output: (batch_size, 1280) -> (batch_size, output_size)
		preds = self.fc_out(new_context_vec)
		
		return preds