import sys
import os
from tkinter import *
import tkFileDialog
from PIL import Image, ImageTk
#from cnn import get_data, conv2d_block, get_unet, plot_sample

window = Tk()

window.title("Mushroom ML GUI")
window.geometry("600x600")
window.configure(background="light green")

filename = ""


def set_filename(name):
    global filename
    filename = name
    return


def runCNN():
    os.system("python cnn.py %s" % (filename))
    img = ImageTk.PhotoImage(Image.open(
        "result.png").resize((200, 200), Image.ANTIALIAS))
    panel = Label(window, image=img)
    panel.config(height=200, width=200)
    panel.photo = img
    panel.place(x=350, y=60, height=200, width=200)
    return


def runFFN():
    os.system("python FFN.py %s" % (filename))
    return


def browsefunc():
    name = tkFileDialog.askopenfilename(
        filetypes=[("JPG FILES", "*.jpg"), ("PNG FILES", "*.png")]
    )
    set_filename(name)
    file_field.insert(0, name)
    img = ImageTk.PhotoImage(Image.open(
        filename).resize((200, 200), Image.ANTIALIAS))
    panel = Label(window, image=img)
    panel.config(height=200, width=200)
    panel.photo = img
    panel.place(x=50, y=60, height=200, width=200)
    return


# create labels for photos
Input = Label(window, text="Input File", bg="light green")
Input.place(x=120, y=20)
Input = Label(window, text="Generated Mask", bg="light green")
Input.place(x=400, y=20)

# create Upload File Section
file = Label(window, text="File", bg="light green")
file.place(x=20, y=300)
file_field = Entry(window)
file_field.place(x=70, y=300, width=400)
browsebutton = Button(window, text="Browse", command=browsefunc)
browsebutton.place(x=500, y=300)

# create choose algorithm Section
algorithm = Label(window, text="Algorithm", bg="light green")
algorithm.place(x=20, y=400)
cnn_btn = Button(window, text="CNN", bg="black", command=runCNN)
cnn_btn.place(x=150, y=400, width=120, height=25)
ffn_btn = Button(window, text="FFN", bg="black", command=runFFN)
ffn_btn.place(x=350, y=400, width=120, height=25)

# display team logo
mushroom_img = ImageTk.PhotoImage(
    Image.open("./mushroom.jpg").resize((50, 50), Image.ANTIALIAS)
)
logo = Label(window, image=mushroom_img)
logo.config(height=50, width=50)
logo.photo = mushroom_img
logo.place(x=0, y=550, height=50, width=50)

window.mainloop()
