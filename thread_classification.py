#!/usr/bin/python3
#
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditi./my-imagenet --log-level=verbose --headless --camera=/dev/video0  --model=/home/nano/i375x1kh/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=/home/nano/i375x1kh/labelsons:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,# print out the result

# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
from PyQt5.QtCore import QThread
import jetson.inference
import jetson.utils
import RPi.GPIO as GPIO
from time import sleep


# Pin Definitions
class_pin = [12, 18, 16]  # BCM pin 18, BOARD pin 12

class Thread_Classification(QThread):
	def __init__(self,model_dir,source_video,parent=None):
		super(Thread_Classification, self).__init__(parent)
		print('Begin : classification, can take a long time to start')
		sleep(0.01)
		self.net = jetson.inference.imageNet(argv=['thread_classification.py', '--model='+model_dir+'/resnet18.onnx', '--labels='+model_dir+'/labels.txt', '--input_blob=input_0', '--output_blob=output_0'])  #
		# create video sources & outputs
		self.source_video = source_video
		# Pin Setup:
		GPIO.setmode(GPIO.BOARD)  # BCM pin-numbering scheme from Raspberry Pi
		# set pin as an output pin with optional initial state of HIGH
		for pin in class_pin:
			GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
		self.classification_OK = True

	def run(self):
		try:
			# process frames until the user exits
			while self.classification_OK:
				# capture the next image
				img = self.source_video.Capture()
				# classify the image
				class_id, confidence = self.net.Classify(img)
				# Switch on/off the LEDs
				for index, pin in enumerate(class_pin):
					GPIO.output(pin, class_id==index)
				# find the object description
				class_desc = self.net.GetClassDesc(class_id)
				# print out the result
				print("image is recognized as '{:s}' (class #{:d}) with {:f}% confidence".format(class_desc, class_id, confidence * 100))
				sleep(1)  # One inference per second
		finally:
			for pin in class_pin:
				GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
			GPIO.cleanup()

if __name__ == '__main__':
 	cls = Thread_Classification('Projects/good/model')
 	cls.start()


