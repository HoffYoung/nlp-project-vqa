import mindspore.nn as nn
import mindspore.ops.operations as P
from mindspore import dtype as mstype
from mindspore import Tensor


# class DummyAccuracyMetric(nn.Metric):
# 	def __init__(self):
# 		super(DummyAccuracyMetric, self).__init__()
# 		self.clear()
# 	def clear(self):
# 		self._correct_num = 0
# 		self._total_num = 0
# 	def update(self, inputs):
# 		return 
# 	def eval(self):
# 		return 1.0


class Accuracy(nn.Cell):
	def __init__(self):
		super(Accuracy, self).__init__()
		self.equal = P.Equal()
		self.argmax = P.Argmax(axis=-1)
		self.cast = P.Cast()
		self.sum = P.ReduceSum()

	def construct(self, preds, answers):
		preds = self.cast(preds, mstype.float32)
		correct_prediction = self.equal(self.argmax(preds), answers)
		accuracy_num = self.cast(correct_prediction, mstype.float32)
		return self.sum(accuracy_num) / accuracy_num.shape[0]


if __name__ == "__main__":
	x = [[1,2,3],[2,3,2],[0,1,2]]
	y = [0,1,2]
	x = Tensor(x)
	y = Tensor(y)

	acc = Accuracy()
	print(acc(x, y))