'''
This file is written by SEGP Group 7b
This file is to determine the illumination signal based on the result obtained to output the signal to light up laser diode
'''

import RPi.GPIO as GPIO
import os
import time
import numpy as np
import re

#Set GPIO mode to utilize raspberry pi pins
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)

#Initialize variables
cathodes = [7,11]
anodes = [12,16]
sleeptime = 0.001
timer = 0
delay = 10000


#Laser function that specifies what to display on the laser illumination system
#Uses multiplexing to control the lasers
def laser():
    global timer
    running = True
    
    #Assign the cathode pins to cathodes array and setup pins
    for cathode in cathodes:
        GPIO.setup(cathode, GPIO.OUT)
        GPIO.output(cathode, 0)

    #Assign the anode pins to anodes array and setup pins    
    for anode in anodes:
        GPIO.setup(anode, GPIO.OUT)
        GPIO.output(anode, 0)


    #Loop the laser illumination to be always displaying while main system is running
    while running:
        
        #While looping, read the latest laser_index.txt file which contains the latest output from running the seed prediction
        f = open('/home/pi/Desktop/PI_SEGP/resources/temp/laser_index.txt', 'r')
        str_matrix = f.readline()
        
        try:
            #Convert matrix string to integer list (array)
            matrix = str2matrix(str_matrix)
        except:
            print("String index out of range")
        

        #Only 1 column can display at a time due to multiplexing
        #Cycle through columns with 0.001s delay to display seamless transition between columns
        for i in range(2):
            GPIO.output(cathodes[0], matrix[i][0])
            GPIO.output(cathodes[1], matrix[i][1])
            
            #Update current column
            GPIO.output(anodes[i], 1)
            
            #Set delay to 0.001s
            time.sleep(sleeptime)
            
            #Disconnect current column
            GPIO.output(anodes[i], 0)
        
        #Check if system is still running and update running
        f = open('/home/pi/Desktop/PI_SEGP/resources/temp/running_status.txt', 'r')
        running = f.readline()
        
        #Break from loop if running is false
        if running == "False":
           break

    GPIO.cleanup()

#Converts matrix string eg: "[1, 1]" to integer array [1,1]
def str2matrix(str_matrix):
    matrix = [[1,1],[1,1]]
    matrix[0][0] = int(str_matrix[2])
    matrix[0][1] = int(str_matrix[5])
    matrix[1][0] = int(str_matrix[10])
    matrix[1][1] = int(str_matrix[13])
     
    for i in range(2):
        for j in range(2):
            if matrix[i][j] == 1:
                matrix[i][j] = 0
            elif matrix[i][j] == 0:
                matrix[i][j] = 1
                

    return matrix
    
