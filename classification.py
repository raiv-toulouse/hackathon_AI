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

import jetson.inference
import jetson.utils
import RPi.GPIO as GPIO
import time


# Pin Definitions
class_0_pin = 12  # BCM pin 18, BOARD pin 12
class_1_pin = 16  
class_2_pin = 18  

class Classification:
	def __init__(self,model_dir):
		self.net = jetson.inference.imageNet(argv=['classification.py', '--model='+model_dir+'/resnet18.onnx', '--labels='+model_dir+'/labels.txt', '--input_blob=input_0', '--output_blob=output_0', 'csi://0'])  # ,'--input_blob=input_0'
		# create video sources & outputs
		self.input = jetson.utils.videoSource("csi://0")
		# Pin Setup:
		GPIO.setmode(GPIO.BOARD)  # BCM pin-numbering scheme from Raspberry Pi
		# set pin as an output pin with optional initial state of HIGH
		GPIO.setup(class_0_pin, GPIO.OUT, initial=GPIO.LOW)
		GPIO.setup(class_1_pin, GPIO.OUT, initial=GPIO.LOW)
		GPIO.setup(class_2_pin, GPIO.OUT, initial=GPIO.LOW)

	def classify(self):
		try:
			# process frames until the user exits
			while True:
				# capture the next image
				img = self.input.Capture()
				# classify the image
				class_id, confidence = self.net.Classify(img)
				# Switch on/off the LEDs
				GPIO.output(class_0_pin, class_id==0)
				GPIO.output(class_1_pin, class_id==1)
				GPIO.output(class_2_pin, class_id==2)
				# find the object description
				class_desc = self.net.GetClassDesc(class_id)
				# print out the result
				print("image is recognized as '{:s}' (class #{:d}) with {:f}% confidence".format(class_desc, class_id, confidence * 100))
				time.sleep(1)  # One inference per second
		finally:
			GPIO.cleanup()
	

cls = Classification('tyty/model')
cls.classify()


