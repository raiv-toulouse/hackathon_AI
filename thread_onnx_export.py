#
# converts a saved PyTorch model to ONNX format
# 
import os
from PyQt5.QtCore import QThread,QTimer,pyqtSignal
import torch
import torchvision.models as models
from reshape import reshape_model
from time import sleep


class Thread_ONNX_export(QThread):
	signalEndExport = pyqtSignal()

	def __init__(self, model_dir, parent=None):
		super(Thread_ONNX_export, self).__init__(parent)
		self.model_dir = os.path.expanduser(model_dir)
		self.input = os.path.join(model_dir, 'model_best.pth.tar') # path to input PyTorch model
		self.no_softmax = False # disable adding nn.Softmax layer to model (default is to add Softmax)

	def run(self):
		# set the device
		device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		print('running on device ' + str(device))
		# load the model checkpoint
		print('loading checkpoint:  ' + self.input)
		checkpoint = torch.load(self.input)
		arch = checkpoint['arch']
		# create the model architecture
		print('using model:  ' + arch)
		model = models.__dict__[arch](pretrained=True)
		sleep(0.01)
		# reshape the model's output
		model = reshape_model(model, arch, checkpoint['num_classes'])
		# load the model weights
		model.load_state_dict(checkpoint['state_dict'])
		# add softmax layer
		if not self.no_softmax:
			print('adding nn.Softmax layer to model...')
			model = torch.nn.Sequential(model, torch.nn.Softmax(1))
		model.to(device)
		model.eval()
		print(model)
		sleep(0.01)
		# create example image data
		resolution = checkpoint['resolution']
		input = torch.ones((1, 3, resolution, resolution)).cuda()
		print('input size:  {:d}x{:d}'.format(resolution, resolution))
		# format output model path
		self.output = arch + '.onnx'
		if self.model_dir and self.output.find('/') == -1 and self.output.find('\\') == -1:
			self.output = os.path.join(self.model_dir, self.output)
		# export the model
		input_names = [ "input_0" ]
		output_names = [ "output_0" ]
		print('exporting model to ONNX...')
		torch.onnx.export(model, input, self.output, verbose=True, input_names=input_names, output_names=output_names)
		print('model exported to:  {:s}'.format(self.output))
		self.signalEndExport.emit()


