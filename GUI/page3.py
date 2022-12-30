import tkinter as tk
from PIL import Image, ImageTk
import requests

HEIGHT = 700
WIDTH = 1280
root = tk.Tk()

def nextPage():
    root.destroy()
    import page2

def firstPage():
    root.destroy()
    import page1

canvas = tk.Canvas(root, height=HEIGHT, width=WIDTH)
canvas.pack()

image = Image.open("buttons/page3Next.png")
btnPhoto= ImageTk.PhotoImage(image)

image2 = Image.open("buttons/page3Reset.png")
btnPhoto2= ImageTk.PhotoImage(image2)

background_image = tk.PhotoImage(file='PageImagesFolder/pg3.png')
background_label = tk.Label(root, image=background_image)
background_label.place(relwidth=1, relheight=1)

nextbutton = tk.Button(background_label,command= nextPage,image=btnPhoto,borderwidth=0)
nextbutton.place(x=400,y=580, height=80, width=200)

resetbutton = tk.Button(background_label,command= firstPage,image=btnPhoto2,borderwidth=0)
resetbutton.place(x=100,y=600, height=50, width=50)

root.mainloop()