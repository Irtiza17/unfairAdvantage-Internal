import time
from tkinter import *
from tkinter import messagebox

###create interface
clockWindow = Tk()
clockWindow.geometry('500x500')
clockWindow.title('Countdown Timer')
clockWindow.configure(background='#80c5ef')

hourString = StringVar()
minuteString = StringVar()
secondString = StringVar()

## Set strings to default value
hourString.set('00')    
minuteString.set('00')    
secondString.set('5')    

###get user input

# hourTextbox=Entry(clockWindow,width=3,bg='#80c5ef',font=('Calibri',20,""),textvariable=hourString)
# minuteTextbox=Entry(clockWindow,width=3,font=('Calibri',20,""),textvariable=minuteString)
secondTextbox=Entry(clockWindow,justify=LEFT,font=('Arial',180,"normal"),fg='White',bg ="#80c5ef",textvariable=secondString)

## Center textboxes
# hourTextbox.place(x=170,y=100)
# minuteTextbox.place(x=220,y=100)
secondTextbox.place(x=270,y=100,width=180,height=180)

# def runTimer():
try:
    # clockTime = int(hourString.get())*3600+int(minuteString.get())*60+int(secondString.get())
    clockTime = int(secondString.get())
except:
    print('incorrect value')

# while(clockTime> -1):
#     totalMinutes,totalSeconds = divmod(clockTime,60)
#     totalHours=0
#     if (totalMinutes > 60):
#         totalHours,totalMinutes=divmod(totalMinutes,60)
#     hourString.set("{0:2d}".format(totalHours))
#     minuteString.set("{0:2d}".format(totalMinutes))
#     secondString.set("{0:2d}".format(totalSeconds))

#     ##Update the interface
#     clockWindow.update()
#     time.sleep(1)

#     ## Let the user know if timer has expired
#     # if(clockTime == 0):
#     #     messagebox.showinfo("","Your time has expired!")
#     clockTime -=1

# setTimebutton= Button(clockWindow,text='Set Time',bd='5',command=runTimer)
# setTimebutton.place(relx=0.5,rely=0.5,anchor=CENTER)

###Keep Looping
clockWindow.mainloop()
