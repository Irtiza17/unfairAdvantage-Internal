import tkinter as tk

root = tk.Tk()
root.geometry('1280x726')
root.title('Unfair Advantage')

mainDirectory = '../nahdi01/'

def addDetails():
    bg = tk.PhotoImage(file = mainDirectory +"images/bg/formFill.png")
    label = tk.Label(root,image=bg)
    label.place(x=0,y=0)

    textbox = tk.Text(root,height=1,width=44,font=('Arial',24),pady=16,padx=7)
    textbox.place(x=405,y=186)
 

    btn1_image = tk.PhotoImage(file = mainDirectory + "images/buttons/next.png")
    btn1 = tk.Button(root,image=btn1_image,command=placeYourChild, bd=0, relief='groove',cursor='hand2')
    btn1.place(x=430,y=630)

    btn2_image = tk.PhotoImage(file = mainDirectory + "images/buttons/reset.png")
    btn2 = tk.Button(root,image=btn2_image,command=main, bd=0, highlightthickness = 0, relief='groove',cursor='hand2')
    btn2.place(x=1132,y=650)

    root.mainloop()
    # print('hello')


def placeYourChild():
    bg = tk.PhotoImage(file = mainDirectory +"images/bg/placeYourChild.png")
    label = tk.Label(root,image=bg)
    label.place(x=0,y=0)

    btn1_image = tk.PhotoImage(file = mainDirectory + "images/buttons/next2.png")
    btn1 = tk.Button(root,image=btn1_image,command=placeYourChild, bd=0, relief='groove',cursor='hand2')
    btn1.place(x=430,y=605)

    btn2_image = tk.PhotoImage(file = mainDirectory + "images/buttons/reset.png")
    btn2 = tk.Button(root,image=btn2_image,command=main, bd=0, highlightthickness = 0,cursor='hand2')
    btn2.place(x=1132,y=650)

    root.mainloop()

def main():
    bg = tk.PhotoImage(file = mainDirectory + "images/bg/mainScreen.png")
    label = tk.Label(root,image=bg)
    label.place(x=0,y=0)

    btn1_image = tk.PhotoImage(file = mainDirectory + "images/buttons/english.png")
    btn1 = tk.Button(root,image=btn1_image,command=addDetails, bd=0, relief='groove',cursor='hand2')
    btn1.place(x=935,y=285)
    # btn1.pack()

    root.mainloop()

if __name__ == '__main__':
    main()