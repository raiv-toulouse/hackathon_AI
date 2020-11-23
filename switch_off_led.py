#!/usr/bin/python3

import RPi.GPIO as GPIO

# Pin Definitions
class_0_pin = 12  # BCM pin 18, BOARD pin 12
class_1_pin = 16  
class_2_pin = 18  

GPIO.setwarnings(False)

# Pin Setup:
GPIO.setmode(GPIO.BOARD)  # BCM pin-numbering scheme from Raspberry Pi
# set pin as an output pin with optional initial state of HIGH
GPIO.setup(class_0_pin, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(class_1_pin, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(class_2_pin, GPIO.OUT, initial=GPIO.LOW)

GPIO.cleanup()
