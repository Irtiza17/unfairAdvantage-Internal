from tkinter import *
from tkinter import messagebox
from tkVideoPlayer import TkinterVideo
from PIL import Image,ImageTk
import pymysql
import main
import smtplib
from email.message import EmailMessage
from globalFuncs import *
from threading import Thread


video_file_name = "demovideo.mp4"
video_display = True
countdown = '5'

class unfairGUI:

    def __init__(self,root,videoFile,displayVideo,countdown):
        self.videofile = videoFile
        self.sec = countdown
        self.displayVideo = displayVideo
        self.con=pymysql.connect(host='localhost',user='root',passwd='1234',port=3307,database='unfair_data')
        self.root = root
        self.root.title('Screen 1')
        self.root.geometry('1280x756')
        self.root.config()
        self.screen3()

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

    def click(self,event,placeholder):
        if event.get() == placeholder:
            event.config(state=NORMAL)
            event.delete(0, 'end')

    def leave(self,event,placeholder):
        # event.delete(0, 'end')
        if event.get() == "":
            event.insert(0, f'{placeholder}')
            event.config(state=DISABLED)
            self.root.focus()

    def screen2(self):

        self.root.title('Screen 2')
        #Form Frame
        self.bgScreen2 = ImageTk.PhotoImage(file = 'images/bg/pg2.png')
        scrn2Label = Label(self.root,image=self.bgScreen2).place(x=0,y=0)
        # frame1 = Frame(self.root,width=812,height=402)
        # frame1.place(x=402,y=184)
        # frame1.place(relx=20,rely=20,width=812,height=402)

        #Field1
        self.txt_mname = Entry(self.root,font=('calibri',24),bg='white',bd=0)
        self.txt_mname.insert(0, 'Mother Name')
        self.txt_mname.config(state=DISABLED)
        self.txt_mname.bind("<Button-1>", lambda e : self.click(self.txt_mname,'Mother Name'))
        self.txt_mname.bind("<Leave>", lambda e : self.leave(self.txt_mname,'Mother Name'))
        self.txt_mname.place(x=412,y=188,width=790,height=65)
        

        #Field 2
        self.txt_cname = Entry(self.root,font=('calibri',24),bg='white',bd=0)
        self.txt_cname.insert(0, "Child's Name")
        self.txt_cname.config(state=DISABLED)
        self.txt_cname.bind("<Button-1>", lambda e : self.click(self.txt_cname,"Child's Name"))
        self.txt_cname.bind("<Leave>", lambda e : self.leave(self.txt_cname,"Child's Name"))
        self.txt_cname.place(x=412,y=(188+82),width=790,height=65)

        #Field 3
        self.txt_cage = Entry(self.root,font=('calibri',24),bg='white',bd=0)
        self.txt_cage.insert(0, "Child's Age")
        self.txt_cage.config(state=DISABLED)
        self.txt_cage.bind("<Button-1>", lambda e : self.click(self.txt_cage,"Child's Age"))
        self.txt_cage.bind("<Leave>", lambda e : self.leave(self.txt_cage,"Child's Age"))
        self.txt_cage.place(x=412,y=(188+164),width=380,height=65)

        #Field 4
        self.txt_cgender = Entry(self.root,font=('calibri',24),bg='white',bd=0)
        self.txt_cgender.insert(0, "Child's Gender")
        self.txt_cgender.config(state=DISABLED)
        self.txt_cgender.bind("<Button-1>", lambda e : self.click(self.txt_cgender,"Child's Gender"))
        self.txt_cgender.bind("<Leave>", lambda e : self.leave(self.txt_cgender,"Child's Gender"))
        self.txt_cgender.place(x=822,y=(188+164),width=380,height=65)

        #Field 5
        self.txt_email = Entry(self.root,font=('calibri',24),bg='white',bd=0)
        self.txt_email.insert(0, "Email")
        self.txt_email.config(state=DISABLED)
        self.txt_email.bind("<Button-1>", lambda e : self.click(self.txt_email,"Email"))
        self.txt_email.bind("<Leave>", lambda e : self.leave(self.txt_email,"Email"))
        self.txt_email.place(x=412,y=(188+246),width=790,height=65)

        #Field 6c
        self.txt_mob = Entry(self.root,font=('calibri',24),bg='white',bd=0)
        self.txt_mob.insert(0, "Mobile Number")
        self.txt_mob.config(state=DISABLED)
        self.txt_mob.bind("<Button-1>", lambda e : self.click(self.txt_mob,"Mobile Number"))
        self.txt_mob.bind("<Leave>", lambda e : self.leave(self.txt_mob,"Mobile Number"))
        self.txt_mob.place(x=412,y=(188+328),width=790,height=65)

        # Next Button
        self.btn1_image = PhotoImage(file = "images/buttons/next.png")
        btn_next = Button(self.root,image = self.btn1_image,bd = 0, relief='groove',cursor='hand2', highlightthickness = 0,command=self.register_data)
        btn_next.place(x=430,y=630)

        #Reset
        self.btnResetImage = PhotoImage(file ="images/buttons/reset.PNG")
        btn_reset = Button(self.root,image=self.btnResetImage, highlightthickness = 0,command=self.screen3, bd=0, relief='groove',cursor='hand2')
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
        btn_reset = Button(self.root,image=self.btnResetImage, highlightthickness = 0,command=self.screen3, bd=0, relief='groove',cursor='hand2')
        btn_reset.place(x=1132,y=650)
    
    def countdownTimer(self,sec):
        secondString = StringVar()
        secondString.set(f'{sec}')
        secondTextbox=Entry(self.root,justify=LEFT,font=('Calibri Bold',200,"normal"),fg='White',bd=0,bg ="#80c5ef",textvariable=secondString)
        secondTextbox.place(x=760,y=310,width=135,height=200)
        clockTime = int(secondString.get())
        while(clockTime> -1):
            totalMinutes,totalSeconds = divmod(clockTime,60)
            totalHours=0
            if (totalMinutes > 60):
                totalHours,totalMinutes=divmod(totalMinutes,60)
            secondString.set("{0:1d}".format(totalSeconds))

        ##Update the interface
            self.root.update()
            time.sleep(1)
            clockTime -=1 

    def screen4(self): 

        #BG Image
        self.root.title('Screen 4')
        self.bgScreen4 = ImageTk.PhotoImage(file = 'images/bg/pg4.png')
        self.scrn4Label=Label(self.root,image=self.bgScreen4)
        self.scrn4Label.place(x=0,y=0)

        # Next Button
        # self.btn1_image = PhotoImage(file = "images/buttons/next.png")
        # btn_next = Button(self.root,image = self.btn1_image,bd = 0, relief='groove',cursor='hand2', highlightthickness = 0,command=self.screen5)
        # btn_next.place(x=430,y=630)
        
        #Reset
        self.btnResetImage = PhotoImage(file ="images/buttons/reset.PNG")
        btn_reset = Button(self.root,image=self.btnResetImage, highlightthickness = 0,command=self.screen3, bd=0, relief='groove',cursor='hand2')
        btn_reset.place(x=1132,y=650)
        self.countdownTimer(self.sec)
        # self.scrn4Label.after(0,self.assessmentStart)
        self.assessmentStart()

    def emailFunc(self):
        cur2=self.con.cursor()
        cur2.execute("select Email from form_data order by id desc limit 1")
        row = cur2.fetchone()
        email = row[0]
        print("Report Sent to Email : " ,email)        

        # # set your email and password
        # # please use App Password
        # email_address = "my-gmail-address@gmail.com"
        # email_password = "app-password-for-gmail"

        # # create email
        # msg = EmailMessage()
        # msg['Subject'] = "Email subject"
        # msg['From'] = email_address
        # msg['To'] = "to-address@gmail.com"
        # msg.set_content("This is eamil message")

        # # send email
        # with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        #     smtp.login(email_address, email_password)
        #     smtp.send_message(msg)
        self.root.after(4000,self.screen6())
    
    def whatsappFunc(self):
        cur2=self.con.cursor()
        cur2.execute("select Mobile_Number from form_data order by id desc limit 1")
        row = cur2.fetchone()
        number = row[0]
        print("Report Sent to Number : " ,number)
        self.root.after(4000,self.screen6())

    def screen5(self): 
        #BG Image
        self.root.title('Screen 5')
        self.bgScreen5 = ImageTk.PhotoImage(file = 'images/bg/pg5a.png')
        scrn5Label=Label(self.root,image=self.bgScreen5).place(x=0,y=0)

        # Email Button
        self.btnEmail = PhotoImage(file = "images/buttons/email.png")
        btn_email = Button(self.root,image = self.btnEmail,bd = 0, relief='groove',cursor='hand2', highlightthickness = 0,command=self.emailFunc)
        btn_email.place(x=430,y=585)

        # Whatsapp Button
        self.btnWa = PhotoImage(file = "images/buttons/whatsapp.png")
        btn_wa = Button(self.root,image = self.btnWa,bd = 0, relief='groove',cursor='hand2', highlightthickness = 0,command=self.whatsappFunc)
        btn_wa.place(x=508,y=583)
        

        #Reset
        self.btnResetImage = PhotoImage(file ="images/buttons/reset.PNG")
        btn_reset = Button(self.root,image=self.btnResetImage, highlightthickness = 0,command=self.screen3, bd=0, relief='groove',cursor='hand2')
        btn_reset.place(x=1132,y=650)

    def screen6(self): 
        #BG Image
        self.root.title('Screen 6')
        self.bgScreen6 = ImageTk.PhotoImage(file = 'images/bg/pg6.png')
        scrn6Label=Label(self.root,image=self.bgScreen6)
        scrn6Label.place(x=0,y=0)
        #Reset
        self.btnResetImage = PhotoImage(file ="images/buttons/reset.PNG")
        btn_reset = Button(self.root,image=self.btnResetImage, highlightthickness = 0,command=self.screen3, bd=0, relief='groove',cursor='hand2')
        btn_reset.place(x=1132,y=650)
        scrn6Label.after(10000, self.screen7)

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
        btn_reset = Button(self.root,image=self.btnResetImage, highlightthickness = 0,command=self.screen3, bd=0, relief='groove',cursor='hand2')
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
        btn_reset = Button(self.root,image=self.btnResetImage, highlightthickness = 0,command=self.screen3, bd=0, relief='groove',cursor='hand2')
        btn_reset.place(x=1132,y=650)

    def screen9(self):
        #BG Image
        self.root.title('Screen 9')
        self.bgScreen9 = ImageTk.PhotoImage(file = 'images/bg/pg5b.png')
        scrn9Label=Label(self.root,image=self.bgScreen9)
        scrn9Label.place(x=0,y=0)
        #Reset
        self.btnResetImage = PhotoImage(file ="images/buttons/reset.PNG")
        btn_reset = Button(self.root,image=self.btnResetImage, highlightthickness = 0,command=self.screen3, bd=0, relief='groove',cursor='hand2')
        btn_reset.place(x=1132,y=650)

        #Restart2
        self.btnRestart2Image = PhotoImage(file ="images/buttons/restart2.PNG")
        btn_restart2 = Button(self.root,image=self.btnRestart2Image, highlightthickness = 0,command=self.screen1, bd=0, relief='groove',cursor='hand2')
        btn_restart2.place(x=520,y=440)
  
    def videoPlayer(self):
            vidplayer = TkinterVideo(self.root,scaled=True)
            vidplayer.pack(expand=True,fill='both')
            filepath = f'videos/{self.videofile}'
            vidplayer.load(filepath)
            vidplayer.play()
            self.root.mainloop()
            vidplayer.destroy()
            
    def assessmentStart(self):
        try:
            # self.videoPlayer()
            # thread = Thread(target = self.videoPlayer)
            # thread.start()
            # thread.join()
            videoProcess = videoDisplayFunc(self.videofile,self.displayVideo)
            # self.videoPlayer()
            score, intelligence = main.main()
            cur=self.con.cursor()
            cur.execute("insert into form_data (Score,Intelligence_Type) values(%s,%s)",
                                (score,
                                intelligence
                                ))
            self.con.commit()
            self.con.close()

            videoProcess.terminate() if videoProcess != None else print('No video played')
            self.screen5()

        except Exception as e:
            print(e)
            self.screen9()

    def clear(self):
        self.txt_mname.delete(0,END)
        self.txt_cname.delete(0,END)
        self.txt_cage.delete(0,END)
        self.txt_cgender.delete(0,END)
        self.txt_email.delete(0,END)
        self.txt_mob.delete(0,END)

    def register_data(self):
        
        if self.txt_mname.get() == "Mother Name" or self.txt_cname.get() == "Child's Name" or self.txt_cage.get() == "Child's Age" or self.txt_cgender.get() == "Child's Gender" or self.txt_email.get() == "Email" or self.txt_mob.get() == "Mobile Number":
            messagebox.showerror("Error","All fields are required",parent =self.root)
        else:
            try:
            # mydb = mysql.connector.connect(
            #     host ="localhost",
            #     port = "3307",
            #     username = "root",
            #     password ="1234"
            # )
            # print(mydb)
                cur=self.con.cursor()
                cur.execute("select * from form_data where Email=%s",self.txt_email.get())
                row_email= cur.fetchone()
                cur.execute("select * from form_data where Child_Name=%s",self.txt_cname.get())
                row_child= cur.fetchone()
                if row_email and row_child != None:
                    messagebox.showerror('Error',f'Child Already Registered',parent =self.root)
                else:
                    cur.execute("insert into form_data (Mother_Name,Child_Name,Child_Age,Child_Gender,Email,Mobile_Number) values(%s,%s,%s,%s,%s,%s)",
                                (self.txt_mname.get(),
                                self.txt_cname.get(),
                                self.txt_cage.get(),
                                self.txt_cgender.get(),
                                self.txt_email.get(),
                                self.txt_mob.get()
                                ))
                    self.con.commit()
                    self.con.close()
                    messagebox.showinfo('Success',f'Data registered',parent =self.root)
                    self.clear()
                    self.screen3()
            except Exception as e:
                messagebox.showerror('Error',f'Error due to : {str(e)}',parent =self.root)
        # self.screen3()

   

if __name__=="__main__":
    root =Tk()
    obj = unfairGUI(root,video_file_name,video_display,countdown)
    root.mainloop()