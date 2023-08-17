import utils

if __name__ == "__main__":
	print("start")
	lab_nn=utils.TUNING()
	lab_nn.lr_tuning()
	lab_nn.decay_tuning()
	lab_nn.loss_function_tuning()
	lab_nn.beta1_tuning()
	lab_nn.beta2_tuning()
	lab_nn.epoch_tuning()
	lab_nn.batch_tuning()
	lab_nn.best_parameters()



