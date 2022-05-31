import mindspore.nn as nn
from .metric_utils import Accuracy

class LossAndAccWrapper(nn.Cell):
	def __init__(self, network, train_config):
		super(LossAndAccWrapper, self).__init__()
		self.network = network
		self.acc = Accuracy()
		self.loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
		self.optimizer = nn.Adam(network.trainable_params(), learning_rate=train_config.lr, beta1=0.9, beta2=0.98)
	
	def construct(self, images, questions, answers):
		outputs = self.network(images, questions)
		loss = self.loss(outputs, answers)
		accuracy = self.acc(outputs, answers)
		return loss, accuracy