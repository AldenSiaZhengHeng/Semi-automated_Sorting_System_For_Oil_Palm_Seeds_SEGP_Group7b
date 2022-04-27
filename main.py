'''
This file is written by SEGP Group 7b
This is the main file to run
This file is contain GUI interface function and collaborate with other file
'''

import label as label

import cv2
import csv
import glob
import tkinter as tk
import os
import time
import pathlib
from tkinter import *
from tkinter import messagebox, ttk
import threading
from ir_sensor import *
from camera import *
from illumination import *
import sys


from PIL import Image, ImageTk

from seed_detection import run_seed_detection

#Initialize Tkinter GUI
root = tk.Tk()
w, h = root.winfo_screenwidth(), root.winfo_screenheight() - 100
root.geometry("%dx%d+0+0" % (w, h))
root.attributes('-fullscreen', False)
root.title("AAResources Sorting System")
# root.resizable(0, 0)
root.configure(background='#9ABDA7')
root.grid_columnconfigure(5, minsize=200)


#Initialize flags/counter variables
running = False
index = 0
starting = True
sleeping = 0
release = 0

#Initialize the size and configuration of the GUI grid
Grid.rowconfigure(root, index=0, weight=1)
Grid.rowconfigure(root, index=1, weight=1)
Grid.rowconfigure(root, index=2, weight=1)
Grid.rowconfigure(root, index=3, weight=1)
Grid.rowconfigure(root, index=4, weight=1)
Grid.rowconfigure(root, index=5, weight=1)
Grid.rowconfigure(root, index=6, weight=1)
Grid.rowconfigure(root, index=7, weight=1)
Grid.columnconfigure(root, index=0, weight=1)
Grid.columnconfigure(root, index=1, weight=1)
Grid.columnconfigure(root, index=2, weight=1)
Grid.columnconfigure(root, index=3, weight=1)
Grid.columnconfigure(root, index=4, weight=1)

#Opens a spreadsheet with the output results tabulated
def openCsv():
    top = tk.Toplevel()
    top.title("CSV file")
    TableMargin = Frame(top, width=500)
    TableMargin.pack(side=TOP)
    scrollbarx = Scrollbar(TableMargin, orient=HORIZONTAL)
    scrollbary = Scrollbar(TableMargin, orient=VERTICAL)
    tree = ttk.Treeview(TableMargin, columns=("Tray ID", "Number Of Good Seeds", "Number Of Bad Seeds", "Date and Time"), height=400,
                        selectmode="extended",
                        yscrollcommand=scrollbary.set, xscrollcommand=scrollbarx.set)
    scrollbary.config(command=tree.yview)
    scrollbary.pack(side=RIGHT, fill=Y)
    scrollbarx.config(command=tree.xview)
    scrollbarx.pack(side=BOTTOM, fill=X)
    tree.heading('Tray ID', text="Tray ID", anchor=W)
    tree.heading('Number Of Good Seeds', text="Number Of Good Seeds", anchor=W)
    tree.heading('Number Of Bad Seeds', text="Number Of Bad Seeds", anchor=W)
    tree.heading('Date and Time', text="Date and Time", anchor=W)
    tree.column('#0', stretch=NO, minwidth=0, width=0)
    tree.column('#1', stretch=NO, minwidth=0, width=200)
    tree.column('#2', stretch=NO, minwidth=0, width=200)
    tree.column('#3', stretch=NO, minwidth=0, width=200)
    tree.column('#4', stretch=NO, minwidth=0, width=200)
    tree.pack()

    with open('out/results.csv') as f:
        reader = csv.DictReader(f, delimiter=',')
        for row in reader:
            tray_id = row['tray_id']
            number_of_good_seeds = row['number_of_good_seeds']
            number_of_bad_seeds = row['number_of_bad_seeds']
            date_time = row['date_time']
            tree.insert("", 0, values=(tray_id, number_of_good_seeds, number_of_bad_seeds, date_time))
            #tray_id, number_of_good_seeds, number_of_bad_seeds, date_time

    top.mainloop()

#Main function to control the flow of the system and call the other operations when necessary
def mainFunction():
    global running
    # if (image_taken = true), get new image file path
    # input will always be an image
    if running:
        #Read the current image input based on a queue system. Each new image input are named after the current tracking number.
        dirname = os.path.join(os.path.dirname(__file__), "resources/temp/current_input")
        f = open('resources/temp/tracking_number.txt', 'r')

        #
        filepath = os.path.join(dirname, f.readline() + '.JPG')

        path = pathlib.PurePath(filepath)
        filename = path.name
        
        #If operation is not paused/stopped, check if the IR sensor detects an object.
        if paused == False:
            #If IR sensor detects an object, call takeImage function and pass the filename parameter to name the image taken
            if Sensor() == 1:
                takeImage(filename)
        
        if not os.path.exists(filepath):
            root.after(100, mainFunction)

        elif os.path.isfile(filepath):
            try:
                output_matrix = run_seed_detection(filename)
                
                output_list.append("out/results/" + filename)
                #Name of the processed image
                output_image_file = filename
                fname = output_image_file

                lst.insert(tk.END, fname)
                lst.bind("<<ListboxSelect>>", display_img)

                os.remove(filepath)
                fileint = int(filename.split('.')[0])
                fileint += 1
                file = str(fileint)
                filepath = os.path.join(dirname, file + ".jpg")
                with open('resources/temp/tracking_number.txt', 'w') as f:
                    f.write(str(fileint))
                    
                with open('resources/temp/laser_index.txt', 'w') as f:
                    f.write(str(output_matrix))

                display_img()
                img_status()
                result = True
                root.after(1000, mainFunction)
            except:
                os.remove(filepath)
                messagebox.showerror("Error", "Invalid tray detected.")
                root.after(1000, mainFunction)

#Function to start the system
#Starts the operation of hardware: camera, sensor, laser illumination
#Starts the operation of software: seed_detection, seed_classification
def startProgram():
    global starting
    global running
    global sleeping
    global stop_threads
    global paused
    paused = False
    running = True
    sys_status(True)

    if starting:
        print("starting")
        startThreads(False)
        starting = False
    
    laser_Operation = threading.Thread(target=laser)
    
    with open('resources/temp/running_status.txt', 'w') as f:
            f.write("True")
        #Set laser light to all off
    with open('resources/temp/laser_index.txt', 'w') as f:
        f.write(str([[0,0],[0,0]]))
    #Set running to True 
    laser_Operation.start()
        

#Function to initialize and start threads
def startThreads(stop_threads):
    start_Operation = threading.Thread(target=mainFunction)

    if stop_threads == False:
        start_Operation.start()
        
        
        print("started")
    
#Function to stop/pause all operations
def stopProgram():
    global running
    global starting
    global sleeping
    global paused
    paused = True
    sys_status(False)


#Function to get the filename/tray number of current output image displayed on the GUI
def img_status():
    current_index = index
    #Get filename by separating filename from filepath
    current_image_status = output_list[current_index].split('/')[2]
    #Get filename without extension (.jpg)
    current_image_status = current_image_status.split('.')[0]
    img_index.config(text="Displaying: Tray " + current_image_status)

#Function to get the state of the system (Started or Stopped)
def sys_status(status):
    if status:
        running_status.configure(text="Status: Started")
    elif status == False:
        running_status.configure(text="Status: Stopped")

#Function to display current output image on the GUI
def display_img():
    current_index = index
    file_name = output_list[current_index]
    image = Image.open(file_name)
    resize_img = image.resize((900, 675))
    img = ImageTk.PhotoImage(resize_img)
    label.config(image=img)
    label.image = img

#Function for next_image button
def next_img():
    global index
    current_index = index + 1
    if current_index < len(output_list):
        index += 1
        display_img()
        img_status()
    else:
        #Show error if no next image
        messagebox.showerror("Error", "  End of file: No Next Image        ")

#Function for previous_image button
def prev_img():
    global index
    current_index = index - 1
    if current_index >= 0:
        index -= 1
        display_img()
        img_status()
    else:
        #Show error if no previous image
        messagebox.showerror("Error", "  End of file: No Previous Image        ")

#Function to exit the program
def exitProgram():
    global release
    release = 1
    root.destroy()
    cv2.destroyAllWindows()
    with open('resources/temp/running_status.txt', 'w') as f:
        f.write("False")
    sys.exit()

lst = tk.Listbox(root, height=4, width=50, borderwidth=5, relief="groove")
#Check if system is still running and display its status (Started or Stopped)   
running_status_string = "Status: Stopped"    
running_status = Label(root, text=running_status_string, fg='#214738',height=2, width=55 , font=("Arial", 20, "bold"))
running_status.configure(background='#ECE4DF')
running_status.grid(row=5, column=1, columnspan=3, sticky="nsw", padx=(100,0), ipady=10)
    

# show status
img_index = Label(root, text="Displaying: Nothing", fg='#214738',height=1, width=55, font=("Arial", 20, "bold"))
img_index.configure(background='#ECE4DF')
img_index.grid(row=5, column=1, columnspan=2, sticky="nw", padx=(100,0), pady=(10, 0), ipady=10)


#padding = Label(root, background='#ECE4DF', width=15, height=)
#padding.grid(row=5, column= 2)

# autopen the first image or output
# placeholder for output images before starting
output_list = []
image_placeholder = "resources/graphics/placeholder.jpg"  # output_image_file
image = Image.open(image_placeholder)  # set the directory for the first picture
resize_img = image.resize((900, 675))
img = ImageTk.PhotoImage(resize_img)
label = tk.Label(root, image=img, height=575, width=800)
label.configure(background='#ECE4DF')
label.grid(row=1, column=1, rowspan=4, columnspan=4, sticky="nsew", padx=(100,0), pady=(40,0), ipadx=0, ipady=0)

# next and previous button
nxt_btn = Image.open("resources/graphics/button_next.png")
nxt_btn = ImageTk.PhotoImage(nxt_btn.resize((192, 150)))
nxtimg_label = tk.Label(image=nxt_btn)

prev_btn = Image.open("resources/graphics/button_previous.png")
prev_btn = ImageTk.PhotoImage(prev_btn.resize((192, 150)))
previmg_label = tk.Label(image=prev_btn)

nxtButton = tk.Button(root, image=nxt_btn, height=0, width=0, borderwidth=0, bd=0, highlightthickness=0, background='#ECE4DF', command=next_img)
prevButton = tk.Button(root, image=prev_btn, height=0, width=0, borderwidth=0, bd=0, highlightthickness=0, background='#ECE4DF', command=prev_img)

prevButton.grid(row=5, column=3, sticky="nsew", ipadx=0, ipady=0)
nxtButton.grid(row=5, column=4, sticky="nsew", ipadx=0, ipady=0)

#Set button size for resizing
btnImg_width = 260
btnImg_height = 146
#Read filepath of button images and assign tkinter label
start_btn = Image.open("resources/graphics/button_start.png")
start_btn = ImageTk.PhotoImage(start_btn.resize((btnImg_width, btnImg_height)))
startImg_label = tk.Label(image=start_btn)
stop_btn = Image.open("resources/graphics/button_stop.png")
stop_btn = ImageTk.PhotoImage(stop_btn.resize((btnImg_width, btnImg_height)))
stopImg_label = tk.Label(image=stop_btn)
data_btn = Image.open("resources/graphics/button_data.png")
data_btn = ImageTk.PhotoImage(data_btn.resize((btnImg_width, btnImg_height)))
dataImg_label = tk.Label(image=data_btn)
exit_btn = Image.open("resources/graphics/button_exit.png")
exit_btn = ImageTk.PhotoImage(exit_btn.resize((btnImg_width, btnImg_height)))
exitImg_label = tk.Label(image=exit_btn)

btn_width = 260
btn_height = 146
#Create tkinter buttons with its function
startButton = tk.Button(root, image=start_btn, height=btn_height, width=btn_width, borderwidth=0, bd=0, highlightthickness=0, background='#9ABDA7',
                        command=startProgram)
stopButton = tk.Button(root, image=stop_btn, height=btn_height, width=btn_width, borderwidth=0, bd=0, highlightthickness=0, background='#9ABDA7',
                       command=stopProgram)
dataButton = tk.Button(root, image=data_btn, height=btn_height, width=btn_width, borderwidth=0, bd=0, highlightthickness=0, background='#9ABDA7',
                       command=openCsv)
exitButton = tk.Button(root, image=exit_btn, height=btn_height, width=btn_width, borderwidth=0, bd=0, highlightthickness=0, background='#9ABDA7',
                       command=exitProgram)

logo_height = 250
logo_width = 250
logo = Image.open("resources/graphics/aar_logo.png")
logo = ImageTk.PhotoImage(logo.resize((250,250)))
logo_label = tk.Label(image=logo, height=logo_height, width=logo_width, borderwidth=0, bd=0, background='#9ABDA7')
logo_label.grid(row=1, column=6, sticky="nsew", padx=(0,180))

#Place buttons' positions in Tkinter grid
startButton.grid(row=2, column=6, sticky="new",padx=(0,180))
stopButton.grid(row=3, column=6, sticky="new", padx=(0,180))
dataButton.grid(row=4, column=6, sticky="new", padx=(0,180))
exitButton.grid(row=5, column=6, sticky="new", padx=(0,180))


root.mainloop()

