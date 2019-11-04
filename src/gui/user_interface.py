import sys
import os
from tkinter import *
import tkFileDialog
from PIL import Image, ImageTk

window = Tk()

window.title("Mushroom ML GUI")
window.geometry("600x200")
window.configure(background="light green")

filename = ""


def set_filename(name):
    global filename
    filename = name
    return


def runCNN():
    os.system("python cnn.py %s" % (filename))
    return


def runFFN():
    os.system("python FFN.py %s" % (filename))
    return


def browsefunc():
    name = tkFileDialog.askopenfilename(filetypes=[("JPG FILES", "*.jpg")])
    set_filename(name)
    file_field.insert(0, name)
    img = ImageTk.PhotoImage(Image.open(filename).resize((100, 100), Image.ANTIALIAS))
    panel = Label(window, image=img)
    panel.config(height=100, width=100)
    panel.photo = img
    panel.grid(column=0, row=3)
    return


# create Upload File Section
file = Label(window, text="File", bg="light green")
file.grid(row=1, column=0)
file_field = Entry(window)
file_field.grid(row=1, column=1, ipadx="100")
browsebutton = Button(window, text="Browse", command=browsefunc)
browsebutton.grid(column=2, row=1)

# create choose algorithm Section
algorithm = Label(window, text="Algorithm", bg="light green")
algorithm.grid(column=0, row=2)
cnn_btn = Button(window, text="CNN", bg="black", command=runCNN)
cnn_btn.grid(column=1, row=2)
ffn_btn = Button(window, text="FFN", bg="black", command=runFFN)
ffn_btn.grid(column=2, row=2)

window.mainloop()
