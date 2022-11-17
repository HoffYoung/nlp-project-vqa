import numpy as np
import math
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import Parameter
from mindspore import load_checkpoint
from mindspore import dtype as mstype
from mindspore.ops import operations as P
from mindspore.ops import DynamicGRUV2
from pretrained.embeddings import *
from pretrained.VGG import *

class TextFeatureCell(nn.Cell):
	"""
	GRU to extract question feature
	"""

	def __init__(self, dict, config, gru_hidden_size=512):
		super(TextFeatureCell, self).__init__(auto_prefix=False)
		embedding_table = Tensor(create_embedding_table(dict, config))
		self.embeddings = nn.Embedding(config.vocab_size, 200, embedding_table=embedding_table)
		self.weight_i, self.weight_h, self.bias_i, self.bias_h = \
            self.gru_default_state(input_size=200, hidden_size=gru_hidden_size)
		self.gru = DynamicGRUV2()
		#self.gru = nn.GRU(input_size=200, hidden_size=gru_hidden_size, batch_first=True, dropout=0.1, bidirectional=False)
		self.h0 = Tensor(np.zeros([config.batch_size, gru_hidden_size]).astype(np.float16))
		self.trans = P.Transpose()
		self.cast = P.Cast()

	def gru_default_state(self, input_size, hidden_size):
		'''Weight init for gru cell'''
		stdv = 1 / math.sqrt(hidden_size)
		weight_i = Parameter(Tensor(
			np.random.uniform(-stdv, stdv, (input_size, 3*hidden_size)).astype(np.float16)), 
							name='gru_weight_i')
		weight_h = Parameter(Tensor(
			np.random.uniform(-stdv, stdv, (hidden_size, 3*hidden_size)).astype(np.float16)), 
							name='gru_weight_h')
		bias_i = Parameter(Tensor(
			np.random.uniform(-stdv, stdv, (3*hidden_size)).astype(np.float16)), name='gru_bias_i')
		bias_h = Parameter(Tensor(
			np.random.uniform(-stdv, stdv, (3*hidden_size)).astype(np.float16)), name='gru_bias_h')
		return weight_i, weight_h, bias_i, bias_h

	def construct(self, x):
		# (batch_size, max_length) -> (batch_size, max_length, 200)
		x = self.embeddings(x)
		# (batch_size, max_length, 200) -> (max_length, batch_size, 200)
		x = self.trans(x, (1, 0, 2))
		# cast float32 to float16
		x = self.cast(x, mstype.float16)
		# hidden: (batch_size, hidden_size)
		_, hiddens, _, _, _, _ = self.gru(x, self.weight_i, self.weight_h, self.bias_i, self.bias_h, None, self.h0)
		hidden = hiddens[-1, :, :]
		# cast float16 to float32
		hidden = self.cast(hidden, mstype.float32)
		return hidden


class TopDownAttentionLayer(nn.Cell):
	"""
	Top Down Attention Layer to deal with image features
	"""

	def __init__(self):
		super(TopDownAttentionLayer, self).__init__()
		self.expand_dims = P.ExpandDims()
		self.tile = P.Tile()
		self.concat = P.Concat(axis=2)
		self.fc_attn_nonlin = nn.Dense(2560, 512, weight_init='normal', has_bias=False, activation='tanh')
		self.fc_attn_lin = nn.Dense(512, 1, weight_init='normal', has_bias=False)
		self.softmax = P.Softmax()
		self.trans = P.Transpose()
		self.matmul = P.BatchMatMul()
		self.reshape = P.Reshape()
		self.mul = P.Mul()
		self.sum = P.ReduceSum()

	def construct(self, images, questions):
		# (batch_size, 512) -> (batch_size, 1, 512)
		questions = self.expand_dims(questions, 1)
		# (batch_size, 1, 512) -> (batch_size, K, 512)
		questions = self.tile(questions, (1, images.shape[1], 1))
		# images: (batch_size, K, 2048)
		# attn_weights: (batch_size, 1, K)
		attn_weights = self.fc_attn_lin(self.fc_attn_nonlin(self.concat((images, questions))))
		attn_weights = self.softmax(self.trans(attn_weights, (0, 2, 1)))
		# (batch_size, 1, K) @ (batch_size, K, 2048) -> (batch_size, 1, 2048)
		vectors = self.matmul(attn_weights, images)
		# (batch_size, 1, 2048) -> (batch_size, 2048)
		vectors = self.reshape(vectors, (-1, 2048))

		return vectors


class TopDownAttentionNet(nn.Cell):
	"""
	Bottom Up - Top Down Attention Network
	Based on the paper: Tips and Tricks for Visual Question Answering: Learnings from the 2017 Challenge
	"""

	def __init__(self, vocab_dict, config):
		super(TopDownAttentionNet, self).__init__()
		self.text_net = TextFeatureCell(vocab_dict, config)
		self.VGGnet = VGG(config)
		self.reshape = P.Reshape()
		self.l2Normalize = P.L2Normalize()
		self.topdown_attn = TopDownAttentionLayer()
		self.fc_questions = nn.Dense(512, 512, activation='tanh')
		self.fc_images = nn.Dense(2048, 512, activation='tanh')
		self.mul = P.Mul()
		self.fc_f_t = nn.Dense(512, 300, activation='tanh')
		self.fc_f_i = nn.Dense(512, 2048, activation='tanh')
		self.fc_w_t = nn.Dense(300, config.output_size, has_bias=False)
		self.fc_w_i = nn.Dense(2048, config.output_size, has_bias=False)
		self.sigmoid = P.Sigmoid()
		
	def construct(self, images, questions):
		questions = self.text_net(questions)
		images = self.VGGnet(images)
		# image feature map resize: (batch_size, 14, 14, 512) -> (batch_size, 49, 2048)
		images = self.reshape(images, (-1, 7*7, 2048))
		# l2 normalization for image
		images = self.l2Normalize(images)
		images_vec = self.topdown_attn(images, questions)
		images_vec = self.fc_images(images_vec)
		questions_vec = self.fc_questions(questions)
		# fusion_vec: (batch_size, 512)
		fusion_vec = self.mul(questions_vec, images_vec)
		images_preds = self.fc_w_i(self.fc_f_i(fusion_vec))
		questions_preds = self.fc_w_t(self.fc_f_t(fusion_vec))
		preds = self.sigmoid(images_preds + questions_preds)

		return preds

class TopDownAttentionNetOpAttn(nn.Cell):
	"""
	Bootom Up - Top Down Attention Network with Options Attention
	Based on the paper: Tips and Tricks for Visual Question Answering: Learnings from the 2017 Challenge 
	"""
	
	def __init__(self, vocab_dict, config):
		super(TopDownAttentionNetOpAttn, self).__init__()
		self.text_net = TextFeatureCell(vocab_dict, config)
		self.VGGnet = VGG(config)
		self.reshape = P.Reshape()
		self.l2Normalize = P.L2Normalize()
		self.topdown_attn = TopDownAttentionLayer()
		self.fc_questions = nn.Dense(512, 512, activation='tanh')
		self.fc_images = nn.Dense(2048, 512, activation='tanh')
		self.mul = P.Mul()
		self.fc_f_t = nn.Dense(512, 300, activation='tanh')
		self.fc_f_i = nn.Dense(512, 2048, activation='tanh')
		self.fc_w_t = nn.Dense(300, config.output_size, has_bias=False)
		self.fc_w_i = nn.Dense(2048, config.output_size, has_bias=False)
		self.sigmoid = P.Sigmoid()
		self.embeddings = nn.Embedding(config.vocab_size, 512)
		self.expand_dims = P.ExpandDims()
		self.fc_query = nn.Dense(512, 512, has_bias=False)
		self.fc_key = nn.Dense(512, 512, has_bias=True)
		self.tanh = nn.Tanh()
		self.fc_h = nn.Dense(512, 1, has_bias=True)
		self.softmax = nn.Softmax(axis=1)
		self.matmul = P.BatchMatMul(transpose_a=True)

	def construct(self, images, questions, options):
		questions = self.text_net(questions)
		images = self.VGGnet(images)
		# image feature map resize: (batch_size, 14, 14, 512) -> (batch_size, 49, 2048)
		images = self.reshape(images, (-1, 7*7, 2048))
		# l2 normalization for image
		images = self.l2Normalize(images)
		images_vec = self.topdown_attn(images, questions)
		images_vec = self.fc_images(images_vec)
		questions_vec = self.fc_questions(questions)
		# fusion_vec: (batch_size, 512)
		fusion_vec = self.mul(questions_vec, images_vec)
		
		# Options (batch_size, 10, 512)
		options = self.embeddings(options)
		fusion_vec = self.expand_dims(fusion_vec, 1)
		# (batch_size, 1, 512) + (batch_size, 10, 512) -> (batch_size, 10, 512)
		h = self.tanh(self.fc_query(fusion_vec) + self.fc_key(options))
		# (batch_size, 10, 1)
		p = self.softmax(self.fc_h(h))
		# (batch_size, 10, 1).T @ (batch_size, 10, 512) -> (batch_size, 1, 512)
		new_context_vec = self.matmul(p, options)
		# (batch_size, 1, 512) -> (batch_size, 512)
		new_context_vec = self.reshape(new_context_vec, (-1, 512))
  
		images_preds = self.fc_w_i(self.fc_f_i(new_context_vec))
		questions_preds = self.fc_w_t(self.fc_f_t(new_context_vec))
		preds = self.sigmoid(images_preds + questions_preds)

		return preds

class TopDownAttentionNetFeature(nn.Cell):
	"""
	Bottom Up - Top Down Attention Network
	Based on the paper: Tips and Tricks for Visual Question Answering: Learning from the 2017 Challenge
	"""

	def __init__(self, vocab_dict, config):
		super(TopDownAttentionNetFeature, self).__init__()
		self.text_net = TextFeatureCell(vocab_dict, config)
		self.reshape = P.Reshape()
		self.l2Normalize = P.L2Normalize()
		self.fc_feature = nn.Dense(8192, 2048, activation='tanh')
		self.topdown_attn = TopDownAttentionLayer()
		self.fc_questions = nn.Dense(512, 512, activation='tanh')
		self.fc_images = nn.Dense(2048, 512, activation='tanh')
		self.mul = P.Mul()
		self.fc_f_t = nn.Dense(512, 300, activation='tanh')
		self.fc_f_i = nn.Dense(512, 2048, activation='tanh')
		self.fc_w_t = nn.Dense(300, config.output_size, has_bias=False)
		self.fc_w_i = nn.Dense(2048, config.output_size, has_bias=False)
		self.sigmoid = P.Sigmoid()
		
	def construct(self, images, questions):
		questions = self.text_net(questions)
		# l2 normalization for image
		images = self.l2Normalize(images)
		# (batch_size, 36, 8192) -> (batch_size, 36, 2048)
		images = self.fc_feature(images)
		images_vec = self.topdown_attn(images, questions)
		images_vec = self.fc_images(images_vec)
		questions_vec = self.fc_questions(questions)
		# fusion_vec: (batch_size, 512)
		fusion_vec = self.mul(questions_vec, images_vec)
		images_preds = self.fc_w_i(self.fc_f_i(fusion_vec))
		questions_preds = self.fc_w_t(self.fc_f_t(fusion_vec))
		preds = self.sigmoid(images_preds + questions_preds)

		return preds
  
if __name__ == "__main__":
	network = TopDownAttentionNet(word_dict, train_config)
	questions = Tensor(np.zeros((32, 14), dtype=np.int32))
	images = Tensor(np.zeros((32, 448, 448, 3), dtype=np.int8))
	output = network(images, questions)
	print(output)