import tkinter as tk
from PIL import Image, ImageTk
import requests

HEIGHT = 700
WIDTH = 1280

def nextPage():
    root.destroy()
    root.mainloop()
    import page2

def get_weather():
    print('button clicked')


root = tk.Tk()
root.title('Unfair Advantage')
root.eval('tk::PlaceWindow . center')

frame1 = tk.Frame(root,width = 800, height=1280,bg="Blue")
canvas = tk.Canvas(root, height=HEIGHT, width=WIDTH)
canvas.pack()

image = Image.open("buttons/English.png")
btnPhoto= ImageTk.PhotoImage(image)

background_image = tk.PhotoImage(file='./PageImagesFolder/pg1.png')
background_label = tk.Label(root, image=background_image)
background_label.place(relwidth=1, relheight=1)
button = tk.Button(background_label,command= nextPage,image=btnPhoto,borderwidth=0)
button.place(x=925,y=250, height=90, width=250)

root.mainloop()

