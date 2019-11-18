import sys
import os
from tkinter import *
import tkFileDialog
from PIL import Image, ImageTk
#from cnn import get_data, conv2d_block, get_unet, plot_sample

window = Tk()
window.title("Mushroom ML GUI")
window.geometry("650x600")
window.configure(background="light green")

filename = ""
modelname = ""


def set_filename(name):
    global filename
    filename = name
    return


def set_modelname(name):
    global modelname
    modelname = name
    return


def browseModelFunc():
    name = tkFileDialog.askopenfilename(
        filetypes=[("PYTHON FILES", "*.py")]
    )
    set_modelname(name)
    model_field.insert(0, name)
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
    panel.place(x=75, y=60, height=200, width=200)
    return


def runCNN():
    os.system("python cnn.py %s" % (filename))
    showResult()
    return


def showResult():
    img = ImageTk.PhotoImage(Image.open(
        "result.png").resize((200, 200), Image.ANTIALIAS))
    panel = Label(window, image=img)
    panel.config(height=200, width=200)
    panel.photo = img
    panel.place(x=375, y=60, height=200, width=200)
    return


def showInput():
    img = ImageTk.PhotoImage(Image.open(
        filename).resize((200, 200), Image.ANTIALIAS))
    panel = Label(window, image=img)
    panel.config(height=200, width=200)
    panel.photo = img
    panel.place(x=375, y=60, height=200, width=200)
    return


def runFFN():
    os.system("python FFN.py %s" % (filename))
    showResult()
    return


def runCustom():
    os.system("python" + " " + modelname + " " + "%s" % (filename))
    if (modelname == ""):
        print("No File Found")
    else:
        showResult()
    return


def toggle():
    if toggle_btn.config('relief')[-1] == 'sunken':
        toggle_btn.config(relief="raised")
        showResult()
    else:
        toggle_btn.config(relief="sunken")
        showInput()
    return


# create labels for photos
Input = Label(window, text="Input File", bg="light green")
Input.place(x=130, y=20)
Input = Label(window, text="Generated Mask", bg="light green")
Input.place(x=420, y=20)

# create toggle button
toggle_btn = Button(text="Show Mask", width=12,
                    relief="raised", command=toggle)
toggle_btn.place(x=400, y=260)

# create Upload File Section
file = Label(window, text="Image", bg="light green")
file.place(x=20, y=325)
file_field = Entry(window)
file_field.place(x=100, y=325, width=400)
browsebutton = Button(window, text="Browse", command=browsefunc)
browsebutton.place(x=525, y=325)

# create Upload Model Section
custom = Label(window, text="Custom", bg="light green")
custom.place(x=20, y=400)
model = Label(window, text="Model", bg="light green")
model.place(x=20, y=420)
model_field = Entry(window)
model_field.place(x=100, y=400, width=400)
browsebutton = Button(window, text="Browse", command=browseModelFunc)
browsebutton.place(x=525, y=400)

# create choose algorithm Section
algorithm = Label(window, text="Algorithm", bg="light green")
algorithm.place(x=20, y=500)
cnn_btn = Button(window, text="CNN", bg="black", command=runCNN)
cnn_btn.place(x=150, y=500, width=120, height=25)
ffn_btn = Button(window, text="FFN", bg="black", command=runFFN)
ffn_btn.place(x=300, y=500, width=120, height=25)
custom_btn = Button(window, text="Custom", bg="black", command=runCustom)
custom_btn.place(x=450, y=500, width=120, height=25)

# display team logo
mushroom_img = ImageTk.PhotoImage(
    Image.open("./mushroom.jpg").resize((50, 50), Image.ANTIALIAS)
)
logo = Label(window, image=mushroom_img)
logo.config(height=50, width=50)
logo.photo = mushroom_img
logo.place(x=0, y=550, height=50, width=50)

window.mainloop()
