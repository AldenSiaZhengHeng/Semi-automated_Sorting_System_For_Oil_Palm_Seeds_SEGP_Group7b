'''
This file is written by SEGP Group 7b
This file is to receive the GPIO input and output of Infrared sensor to trigger image capturing
'''


import RPi.GPIO as GPIO
import time



#Initilize variable for the output of sensor to 0
#0 = No object detected
#1 = Object detected
sensor = 0

#Function to read the signal of IR sensor
def Sensor():
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BOARD)

    #Setup GPIO to read pin 3 (input) and 5 (output)
    GPIO.setup(3,GPIO.IN)
    GPIO.setup(5,GPIO.OUT)
    #Read output of IR sensor at PIN(3)
    val = GPIO.input(3)
    
    #No object detected
    if val == 1:
        GPIO.output(5, GPIO.LOW)
        sensor = False

    #Object detected 
    else:
        GPIO.output(5,GPIO.HIGH)
        sensor = True
    
    return sensor
