import csv
import glob
import tkinter as tk
from tkinter import *
from tkinter import messagebox, ttk

from PIL import Image, ImageTk

root = tk.Tk()
# root.geometry("1300x1000")
root.attributes('-fullscreen', True)
root.title("Semi-automated sorting system")
root.resizable(0, 0)

def openCsv():
    top = tk.Toplevel()
    top.title("CSV file")
    TableMargin = Frame(top, width=500)
    TableMargin.pack(side=TOP)
    scrollbarx = Scrollbar(TableMargin, orient=HORIZONTAL)
    scrollbary = Scrollbar(TableMargin, orient=VERTICAL)
    tree = ttk.Treeview(TableMargin, columns=("ID", "Filename", "Batch number", "Date"), height=400,
                        selectmode="extended",
                        yscrollcommand=scrollbary.set, xscrollcommand=scrollbarx.set)
    scrollbary.config(command=tree.yview)
    scrollbary.pack(side=RIGHT, fill=Y)
    scrollbarx.config(command=tree.xview)
    scrollbarx.pack(side=BOTTOM, fill=X)
    tree.heading('ID', text="ID", anchor=W)
    tree.heading('Filename', text="Filename", anchor=W)
    tree.heading('Batch number', text="Batch number", anchor=W)
    tree.heading('Date', text="Date", anchor=W)
    tree.column('#0', stretch=NO, minwidth=0, width=0)
    tree.column('#1', stretch=NO, minwidth=0, width=200)
    tree.column('#2', stretch=NO, minwidth=0, width=200)
    tree.column('#3', stretch=NO, minwidth=0, width=200)
    tree.column('#4', stretch=NO, minwidth=0, width=200)
    tree.pack()

    with open('test.csv') as f:
        reader = csv.DictReader(f, delimiter=',')
        for row in reader:
            id = row['id']
            filename = row['filename']
            batchnumber = row['batchnumber']
            date = row['date']
            tree.insert("", 0, values=(id, filename, batchnumber, date))

    if __name__ == '__main__':
        top.mainloop()

def showimg(e):
    n = lst.curselection()
    fname = lst.get(n)
    image = Image.open(fname)
    resize_img = image.resize((900, 700))
    img = ImageTk.PhotoImage(resize_img)
    label.config(image=img)
    label.image = img
    print(fname)


def next_selection():
    selection_indices = lst.curselection()

    # default next selection is the beginning
    next_selection = 0
    # make sure at least one item is selected
    if len(selection_indices) > 0:
        # Get the last selection, remember they are strings for some reason
        # so convert to int
        last_selection = int(selection_indices[-1])

        # clear current selections
        lst.selection_clear(selection_indices)

        # Make sure we're not at the last item
        if last_selection < lst.size() - 1:
            next_selection = last_selection + 1
        if last_selection == lst.size() - 1:
            messagebox.showerror("Error", "   End of file.        ")
        else:
            lst.activate(next_selection)
            lst.selection_set(next_selection)
            # show image
            n = lst.curselection()
            fname = lst.get(n)
            image = Image.open(fname)
            resize_img = image.resize((900, 700))
            img = ImageTk.PhotoImage(resize_img)
            label.config(image=img)
            label.image = img


def previous_selection():
    selection_indices = lst.curselection()

    # make sure at least one item is selected
    if len(selection_indices) > 0:
        # Get the last selection, remember they are strings for some reason
        # so convert to int
        last_selection = int(selection_indices[-1])
        # clear current selections
        lst.selection_clear(selection_indices)
        # Make sure we're not at the last item
        if last_selection < lst.size() - 1:
            previous_selection = last_selection - 1
    lst.activate(previous_selection)
    lst.selection_set(previous_selection)
    # show image
    n = lst.curselection()
    fname = lst.get(n)
    image = Image.open(fname)
    resize_img = image.resize((900, 700))
    img = ImageTk.PhotoImage(resize_img)
    label.config(image=img)
    label.image = img


lst = tk.Listbox(root, height=4, width=50)
lst.place(x=390, y=800)

# display the list of file from directory (where the output saved)
namelist = [i for i in glob.glob("*.jpg")]  # set the directory for the output file
for fname in namelist:
    lst.insert(tk.END, fname)
lst.bind("<<ListboxSelect>>", showimg)

# autopen the first image or output
image = Image.open("Line_Mix (s1).jpg")  # set the directory for the first picture
resize_img = image.resize((900, 700))
img = ImageTk.PhotoImage(resize_img)
label = tk.Label(root, text="hello", image=img, height=800, width=1200)
label.place(x=0, y=0)

# next and previous button
nxt_btn = tk.PhotoImage(file="next.png")
nxtimg_label = tk.Label(image=nxt_btn)
prev_btn = tk.PhotoImage(file="previous.png")
previmg_label = tk.Label(image=prev_btn)
nxtButton = tk.Button(root, image=nxt_btn, height=45, width=150, borderwidth=0, command=next_selection)
prevButton = tk.Button(root, image=prev_btn, height=45, width=150, borderwidth=0, command=previous_selection)

prevButton.place(x=150, y=900)
nxtButton.place(x=930, y=900)

# start stop & data button
start_btn = tk.PhotoImage(file="start.png")
startImg_label = tk.Label(image=start_btn)
stop_btn = tk.PhotoImage(file="stop.png")
stopImg_label = tk.Label(image=stop_btn)
data_btn = tk.PhotoImage(file="data.png")
dataImg_label = tk.Label(image=data_btn)
qc_btn = tk.PhotoImage(file="qc.png")
qcImg_label = tk.Label(image=qc_btn)
exit_btn = tk.PhotoImage(file="exit.png")
exitImg_label = tk.Label(image=exit_btn)
startButton = tk.Button(root, image=start_btn, height=100, width=300, borderwidth=0)
stopButton = tk.Button(root, image=stop_btn, height=100, width=300, borderwidth=0)
dataButton = tk.Button(root, image=data_btn, height=100, width=300, borderwidth=0, command=openCsv)
qcButton = tk.Button(root, image=qc_btn, height=100, width=300, borderwidth=0)
exitButton = tk.Button(root, image=exit_btn, height=100, width=300, borderwidth=0, command=root.destroy)

startButton.place(x=1200, y=150)
stopButton.place(x=1200, y=250)
dataButton.place(x=1200, y=350)
qcButton.place(x=1200, y=450)
exitButton.place(x=1200, y=550)

root.mainloop()
