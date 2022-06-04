import numpy as np
import mindspore.ops.operations as P
from mindspore.train.callback import Callback
from mindspore.train.callback import TimeMonitor
from mindspore import save_checkpoint
from mindspore import context

class TrainCallback(Callback):
    def __init__(self, model, valid_dataset, valid_callbacks):
        super(TrainCallback, self).__init__()
        self.model = model
        self.valid_dataset = valid_dataset
        self.valid_callbacks = valid_callbacks
        self.sum = P.ReduceSum()
        self.print = P.Print()
        
    def step_begin(self, run_context):
        if self.valid_callbacks[0].is_early_stop:
            # 提前退出
            run_context.request_stop()

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        # self.print(cb_params)
        epoch_num = cb_params.cur_epoch_num
        step_num = cb_params.cur_step_num
        train_loss = self.sum(cb_params.net_outputs[0]) / cb_params.net_outputs[0].shape[0]
        train_acc = cb_params.net_outputs[1].item(0).asnumpy().item()
        # print(epoch_num)
        # print(step_num)
        if context.get_context('device_target') != 'CPU':
            self.print('epoch:', epoch_num, ', step:', step_num, ' train loss =', train_loss, 'acc =', train_acc)
        else:
            print('epoch:', epoch_num, ', step:', step_num, ' train loss =', train_loss, 'acc =', train_acc)

    def epoch_end(self, run_context):
        # run on valid set
        self.model.eval(self.valid_dataset, callbacks=self.valid_callbacks, dataset_sink_mode=True)


class ValidCallback(Callback):
    def __init__(self, train_config):
        super(ValidCallback, self).__init__()
        self.current_step = 0
        self.valid_acc_max = 0.0
        self.valid_loss_min = np.inf
        self.valid_acc_model = 0
        self.valid_loss_model = np.inf
        self.checkpoint_path = train_config.checkpoint_path
        self.early_stop = train_config.early_stop
        self.is_early_stop = False
        self.sum = P.ReduceSum()
        self.print = P.Print()
    
    def epoch_end(self, run_context):
        # self.print(run_context)
        cb_params = run_context.original_args()
        # self.print(cb_params) 
        valid_loss = self.sum(cb_params.net_outputs[0]) / cb_params.net_outputs[0].shape[0]
        valid_acc = cb_params.net_outputs[1].item(0).asnumpy().item()
        if context.get_context('device_target') != 'CPU':
            self.print('  valid loss =', valid_loss, 'acc =', valid_acc)
        else:
            print('  valid loss =', valid_loss, 'acc =', valid_acc)
        
        if valid_acc >= self.valid_acc_max or valid_loss < self.valid_loss_min:
            if valid_acc >= self.valid_acc_max and valid_loss < self.valid_loss_min:
                self.valid_acc_model = valid_acc
                self.valid_loss_model = valid_loss
                save_checkpoint(cb_params.eval_network.network, self.checkpoint_path)
            self.valid_acc_max = np.max((self.valid_acc_max, valid_acc))
            self.valid_loss_min = np.min((self.valid_loss_min, valid_loss))
            self.current_step = 0
        else:
            self.current_step += 1
            if self.current_step == self.early_stop:
                if context.get_context('device_target') != 'CPU':
                    self.print('early stop... min loss:', self.valid_loss_min, 'max acc:', self.valid_acc_max, end='')
                    self.print('; validation model loss:', self.valid_loss_model, 'acc:', self.valid_acc_model)
                else:
                    print('early stop... min loss:', self.valid_loss_min, 'max acc:', self.valid_acc_max, end='')
                    print('; validation model loss:', self.valid_loss_model, 'acc:', self.valid_acc_model)
                self.is_early_stop = True

class TestCallback(Callback):
    def __init__(self):
        super(TestCallback, self).__init__()
        self.sum = P.ReduceSum()
        self.print = P.Print() if context.get_context('device_target') != 'CPU' else print()
    
    def end(self, run_context):
        cb_params = run_context.original_args()
        self.print(cb_params)
        test_loss = self.sum(cb_params.net_outputs[0]) / cb_params.net_outputs[0].shape[0]
        test_acc = cb_params.net_outputs[1].item(0).asnumpy().item()
        if context.get_context('device_target') != 'CPU':
            self.print('test loss =', test_loss, 'acc =', test_acc)
        else:
            print('test loss =', test_loss, 'acc =', test_acc)

def get_network_callbacks(model, train_dataset, valid_dataset, train_config):
	# eval callbacks
	valid_callback = ValidCallback(train_config)
	valid_callbacks = [valid_callback]
	# train callbacks
	time_callback = TimeMonitor(data_size=train_dataset.get_dataset_size())
	train_callback = TrainCallback(model, valid_dataset, valid_callbacks)
	train_callbacks = [time_callback, train_callback]
	
	return train_callbacks, valid_callbacks