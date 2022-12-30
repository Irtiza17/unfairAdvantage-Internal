from tkinter import *
from tkinter import messagebox
from PIL import Image,ImageTk
import pymysql


class Register:
    def __init__(self,root):
        self.root = root
        self.root.title('Screen 2')
        self.root.geometry('1280x756+0+0')
        self.root.config(bg="white")
        #BG Image
        self.bg = ImageTk.PhotoImage(file = 'images/bg/formFill.png')
        bg=Label(self.root,image=self.bg).place(x=0,y=0)

        #Form Frame
        frame1 = Frame(self.root,bg='white')
        frame1.place(x=402,y=184,width=812,height=402)
        # frame1.place(relx=20,rely=20,width=812,height=402)

        #Field1
        self.txt_mname = Entry(frame1,font=('calibri',24),bg='blue')
        self.txt_mname.place(x=0,y=0,width=812,height=72)
        

        #Field 2
        self.txt_cname = Entry(frame1,font=('calibri',24),bg='blue')
        self.txt_cname.place(x=0,y=82,width=812,height=72)

        #Field 3
        self.txt_cage = Entry(frame1,font=('calibri',24),bg='blue')
        self.txt_cage.place(x=0,y=164,width=400,height=72)

        #Field 4
        self.txt_cgender = Entry(frame1,font=('calibri',24),bg='blue')
        self.txt_cgender.place(x=410,y=164,width=400,height=72)

        #Field 5
        self.txt_email = Entry(frame1,font=('calibri',24),bg='blue')
        self.txt_email.place(x=0,y=246,width=812,height=72)

        #Field 6
        self.txt_mob = Entry(frame1,font=('calibri',24),bg='blue')
        self.txt_mob.place(x=0,y=328,width=812,height=72)

        # Next Button
        self.btn1_image = PhotoImage(file = "images/buttons/next.png")
        btn_next = Button(self.root,image = self.btn1_image,bd = 0, relief='groove',cursor='hand2',command=self.register_data).place(x=430,y=630)


    def register_data(self):
        if self.txt_mname.get() == "" or self.txt_cname.get() == "" or self.txt_cage.get() == "" or self.txt_cgender.get() == "" or self.txt_email.get() == "" or self.txt_mob.get() == "":
            messagebox.showerror("Error","All Fields Are Required",parent =self.root)
        print(self.txt_mname.get()) 

root =Tk()
obj = Register(root)
root.mainloop()