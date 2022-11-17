import mindspore.nn as nn
import mindspore.ops.operations as P
from .metric_utils import Accuracy


class WithLossCellWrapper(nn.Cell):
	def __init__(self, backbone, loss_fn):
		super(WithLossCellWrapper, self).__init__(auto_prefix=False)
		self._backbone = backbone
		self._loss_fn = loss_fn

	def construct(self, images, questions, answers):
		outputs = self._backbone(images, questions)
		return self._loss_fn(outputs, answers)

class WithEvalCellWrapper(nn.Cell):
	def __init__(self, network):
		super(WithEvalCellWrapper, self).__init__(auto_prefix=False)
		self.network = network
		self.softmax = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
		self.acc = Accuracy()
		self.avg = P.ReduceMean()

	def construct(self, images, questions, answers):
		preds = self.network(images, questions)
		accuracy = self.acc(preds, answers)
		preds = self.softmax(preds, answers)
		loss = self.avg(preds)
		return loss, accuracy

class TrainNetworkWrapper(nn.Cell):
	def __init__(self, network, loss_fn, train_config):
		super(TrainNetworkWrapper, self).__init__(auto_prefix=False)
		self.network = network
		loss_net = WithLossCellWrapper(network, loss_fn)
		optimizer = nn.Adam(loss_net.trainable_params(), learning_rate=train_config.lr, weight_decay=train_config.weight_decay, beta1=train_config.momentum, beta2=0.98)
		self._backbone = nn.TrainOneStepCell(loss_net, optimizer)
		self.acc = Accuracy()
		self.avg = P.ReduceMean()
	
	def construct(self, images, questions, answers):
		loss = self._backbone(images, questions, answers)
		loss = self.avg(loss)
		accuracy = self.acc(self.network(images, questions), answers)
		return loss, accuracy