	class SaveCallback(Callback):
    def __init__(self, eval_model, ds_eval):
        super(SaveCallback, self).__init__()
        self.model = eval_model
        self.ds_eval = ds_eval
        self.acc = 0

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        result = self.model.eval(self.ds_eval)
        if result['accuracy'] > self.acc:
            self.acc = result['accuracy']
            file_name = str(self.acc) + ".ckpt"
            save_checkpoint(save_obj=cb_params.train_network, ckpt_file_name=file_name)
            print("Save the maximum accuracy checkpoint,the accuracy is", self.acc)
        if result['loss'] > self.acc:
            self.acc = result['accuracy']
            file_name = str(self.acc) + ".ckpt"
            save_checkpoint(save_obj=cb_params.train_network, ckpt_file_name=file_name)
            print("Save the maximum accuracy checkpoint,the accuracy is", self.acc)
	# model.train(1, train_dataset, ,callbacks=[SaveCallback(model,valid_dataset)], dataset_sink_mode=True)