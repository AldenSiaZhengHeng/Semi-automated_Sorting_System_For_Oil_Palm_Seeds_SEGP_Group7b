*************************************************
Semi-automated Sorting System For Oil Palm Seeds
*************************************************

1. This README file will introduce the files contained in the PI_SEGP folder.
2. There is also a file named SetUp_instruction.pdf which teach user how to set up the whole system with exisiting tools.
3. The explanations have been commented in each file to explain the function to the users.

*******
GitHub
*******
1. The GitHub repository link:
https://github.com/AldenSiaZhengHeng/SEGP_Group7b.git

**********
Important
**********
1. As the folder might run in different devices, please check on the file path in each file whether it is able to match to the directory to read the file required!


***************
PI_SEGP Folder
***************
File created by Group 7b
1. camera.py       - Camera file to capture the images of the tray when triggered by IR sensor.
2. illumination.py - Control the laser diode in illumination system to illuminate the good seeds on tray.
3. ir_sensor.py    - IR sensor file to receive GPIO input and output.
4. main.py         - Main GUI file which contain the main functionality of the system.

Folder created by Group 7b
1. resources folder
   a) Graphics folder          - Contained the graphic images for GUI interface
   b) temp folder
     i)   current_input folder - The folder where captured image stored in.
     ii)  laser_index.txt      - A text file that store the signals in matrix for illumination system
     iii) running_status       - A text file that store the status to tell whether the system is started or stopped
     iv)  tracking_number      - A text file that show the tracking number of the images processed.
     v)   tracking_number_1    - A text file used to reset the tracking_number.txt

2. out folder - Folder that store the output result and read by the system to show it
   a) bbox folder    - current is not used
   b) cropped folder - A folder to store the cropped seed images when performing seed prediction.
   c) results folder - The output result images will be stored in this folder and read by system to show on GUI.
   d) result.csv     - An excel sheet that store the details of each result allows the user to re-tracking the result.

File provided by supervisor, Dr Iman Yi Liao
1. create_csv_file.py
2. preprocess.py
3. seed_classification.py
4. seed_dataset.py
5. seed_detection.py
6. seedxml2.csv.py
7. utils folder
8. src folder
9. models_old folder
10. csv folder
11. __pycache__ folder

*****************************************
Hardware Components Required for Project
*****************************************
1. Infrared (IR) sensor
2. Raspberry Pi 4B 8GB RAM
3. Conveyor belt
4. High Resolution Camera/Webcam
5. Tripod stand
6. illumination system components: laser diode, electronic breadboard, jumper wires and resistors
7. Display Device (laptop, monitor)