from tkinter import *
from tkinter import messagebox
from PIL import Image,ImageTk
import pymysql

class unfairGUI:

    def __init__(self,root):
        self.root = root
        self.root.title('Screen 1')
        self.root.geometry('1280x756')
        self.root.config()
        self.screen8()

    def screen1(self): 

        #BG Image
        self.root.title('Screen 1')
        self.bgScreen1 = ImageTk.PhotoImage(file = 'images/bg/pg1.png')
        scrn1Label=Label(self.root,image=self.bgScreen1).place(x=0,y=0)
        self.btnEngImage = PhotoImage(file ="images/buttons/english.PNG")
        btn_english = Button(self.root,image=self.btnEngImage,command=self.screen2, bd=0,highlightthickness = 0, relief='groove',cursor='hand2')
        btn_english.place(x=935,y=285)

        self.btnArabicImage = PhotoImage(file ="images/buttons/arabic.PNG")
        btn_arabic = Button(self.root,image=self.btnArabicImage, bd=0, highlightthickness = 0,relief='groove',cursor='hand2')
        btn_arabic.place(x=950,y=175)

    def screen2(self):

        self.root.title('Screen 2')
        #Form Frame
        self.bgScreen2 = ImageTk.PhotoImage(file = 'images/bg/pg2.png')
        scrn2Label = Label(self.root,image=self.bgScreen2).place(x=0,y=0)
        frame1 = Frame(self.root,width=812,height=402)
        frame1.place(x=402,y=184)
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
        btn_next = Button(self.root,image = self.btn1_image,bd = 0, relief='groove',cursor='hand2', highlightthickness = 0,command=self.register_data)
        btn_next.place(x=430,y=630)

        #Reset
        self.btnResetImage = PhotoImage(file ="images/buttons/reset.PNG")
        btn_reset = Button(self.root,image=self.btnResetImage, highlightthickness = 0,command=self.screen1, bd=0, relief='groove',cursor='hand2')
        btn_reset.place(x=1132,y=650)

    def screen3(self): 

        #BG Image
        self.root.title('Screen 3')
        self.bgScreen3 = ImageTk.PhotoImage(file = 'images/bg/pg3.png')
        scrn3Label=Label(self.root,image=self.bgScreen3).place(x=0,y=0)

        # Next Button
        self.btn1_image = PhotoImage(file = "images/buttons/next2.png")
        btn_next = Button(self.root,image = self.btn1_image,bd = 0, relief='groove',cursor='hand2', highlightthickness = 0,command=self.screen4)
        btn_next.place(x=430,y=605)

        #Reset
        self.btnResetImage = PhotoImage(file ="images/buttons/reset.PNG")
        btn_reset = Button(self.root,image=self.btnResetImage, highlightthickness = 0,command=self.screen1, bd=0, relief='groove',cursor='hand2')
        btn_reset.place(x=1132,y=650)

    def screen4(self): 

        #BG Image
        self.root.title('Screen 4')
        self.bgScreen4 = ImageTk.PhotoImage(file = 'images/bg/pg4.png')
        scrn4Label=Label(self.root,image=self.bgScreen4)
        scrn4Label.place(x=0,y=0)
        # Next Button
        # self.btn1_image = PhotoImage(file = "images/buttons/next.png")
        # btn_next = Button(self.root,image = self.btn1_image,bd = 0, relief='groove',cursor='hand2', highlightthickness = 0,command=self.screen5)
        # btn_next.place(x=430,y=630)

        #Reset
        self.btnResetImage = PhotoImage(file ="images/buttons/reset.PNG")
        btn_reset = Button(self.root,image=self.btnResetImage, highlightthickness = 0,command=self.screen1, bd=0, relief='groove',cursor='hand2')
        btn_reset.place(x=1132,y=650)
        scrn4Label.after(5000, self.screen5)

    def screen5(self): 
        #BG Image
        self.root.title('Screen 5')
        self.bgScreen5 = ImageTk.PhotoImage(file = 'images/bg/pg5a.png')
        scrn5Label=Label(self.root,image=self.bgScreen5).place(x=0,y=0)

        # Email Button
        self.btnEmail = PhotoImage(file = "images/buttons/email.png")
        btn_email = Button(self.root,image = self.btnEmail,bd = 0, relief='groove',cursor='hand2', highlightthickness = 0,command=self.screen6)
        btn_email.place(x=430,y=585)

        # Whatsapp Button
        self.btnWa = PhotoImage(file = "images/buttons/whatsapp.png")
        btn_wa = Button(self.root,image = self.btnWa,bd = 0, relief='groove',cursor='hand2', highlightthickness = 0,command=self.screen6)
        btn_wa.place(x=508,y=583)
        

        #Reset
        self.btnResetImage = PhotoImage(file ="images/buttons/reset.PNG")
        btn_reset = Button(self.root,image=self.btnResetImage, highlightthickness = 0,command=self.screen1, bd=0, relief='groove',cursor='hand2')
        btn_reset.place(x=1132,y=650)

    def screen6(self): 
        #BG Image
        self.root.title('Screen 6')
        self.bgScreen6 = ImageTk.PhotoImage(file = 'images/bg/pg6.png')
        scrn6Label=Label(self.root,image=self.bgScreen6)
        scrn6Label.place(x=0,y=0)
        #Reset
        self.btnResetImage = PhotoImage(file ="images/buttons/reset.PNG")
        btn_reset = Button(self.root,image=self.btnResetImage, highlightthickness = 0,command=self.screen1, bd=0, relief='groove',cursor='hand2')
        btn_reset.place(x=1132,y=650)
        scrn6Label.after(2000, self.screen7)

    def screen7(self): 
        #BG Image
        self.root.title('Screen 7')
        self.bgScreen7 = ImageTk.PhotoImage(file = 'images/bg/pg7.png')
        scrn7Label=Label(self.root,image=self.bgScreen7)
        scrn7Label.place(x=0,y=0)

        #Restart From the Beginning
        self.btnRestartImage = PhotoImage(file ="images/buttons/Restart.PNG")
        btn_restart = Button(self.root,image=self.btnRestartImage, highlightthickness = 0,command=self.screen8, bd=0, relief='groove',cursor='hand2')
        btn_restart.place(x=660,y=420)

        #Reset
        self.btnResetImage = PhotoImage(file ="images/buttons/reset.PNG")
        btn_reset = Button(self.root,image=self.btnResetImage, highlightthickness = 0,command=self.screen1, bd=0, relief='groove',cursor='hand2')
        btn_reset.place(x=1132,y=650)

    def screen8(self): 
        #BG Image
        self.root.title('Screen 8')
        self.bgScreen8 = ImageTk.PhotoImage(file = 'images/bg/pg8.png')
        scrn8Label=Label(self.root,image=self.bgScreen8)
        scrn8Label.place(x=0,y=0)

        #Restart Confirmation   
        self.btnRestartImage = PhotoImage(file ="images/buttons/Yes.PNG")
        btn_restart = Button(self.root,image=self.btnRestartImage, highlightthickness = 0,command=self.screen1, bd=0, relief='groove',cursor='hand2')
        btn_restart.place(x=477,y=450)

        #Reset
        self.btnResetImage = PhotoImage(file ="images/buttons/reset.PNG")
        btn_reset = Button(self.root,image=self.btnResetImage, highlightthickness = 0,command=self.screen1, bd=0, relief='groove',cursor='hand2')
        btn_reset.place(x=1132,y=650)

    def clear(self):
        self.txt_mname.delete(0,END)
        self.txt_cname.delete(0,END)
        self.txt_cage.delete(0,END)
        self.txt_cgender.delete(0,END)
        self.txt_email.delete(0,END)
        self.txt_mob.delete(0,END)

    def register_data(self):
        
        # if self.txt_mname.get() == "" or self.txt_cname.get() == "" or self.txt_cage.get() == "" or self.txt_cgender.get() == "" or self.txt_email.get() == "" or self.txt_mob.get() == "":
        #     messagebox.showerror("Error","All Fields Are Required",parent =self.root)
        # else:
        #     try:
        #     # mydb = mysql.connector.connect(
        #     #     host ="localhost",
        #     #     port = "3307",
        #     #     username = "root",
        #     #     password ="1234"
        #     # )
        #     # print(mydb)
        #         con=pymysql.connect(host='localhost',user='root',passwd='1234',port=3307,database='unfair_data')
        #         print('connection established : ', con)
        #         cur=con.cursor()
        #         cur.execute("select * from form_data where Email=%s",self.txt_email.get())
        #         row_email= cur.fetchone()
        #         cur.execute("select * from form_data where Child_Name=%s",self.txt_cname.get())
        #         row_child= cur.fetchone()
        #         if row_email and row_child != None:
        #             messagebox.showerror('Error',f'Child Already Registered',parent =self.root)
        #         else:
        #             cur.execute("insert into form_data (Mother_Name,Child_Name,Child_Age,Child_Gender,Email,Mobile_Number) values(%s,%s,%s,%s,%s,%s)",
        #                         (self.txt_mname.get(),
        #                         self.txt_cname.get(),
        #                         self.txt_cage.get(),
        #                         self.txt_cgender.get(),
        #                         self.txt_email.get(),
        #                         self.txt_mob.get()
        #                         ))
        #             con.commit()
        #             con.close()
        #             messagebox.showinfo('Success',f'Data Registered',parent =self.root)
        #             self.clear()
        #             self.screen3()
        #     except Exception as e:
        #         messagebox.showerror('Error',f'Error due to : {str(e)}',parent =self.root)
        self.screen3()

root =Tk()
obj = unfairGUI(root)
root.mainloop()