from tkVideoPlayer import TkinterVideo
from tkinter import *
import vlc
import time


root =Tk()
root.title('Screen 1')
root.geometry('1280x756')

frame = Frame(root, width=700, height=600)
frame.pack()

display = Frame(frame, bd=5)
display.place(relwidth=1, relheight=1)
# vidplayer = TkinterVideo(root,scaled=True)
# vidplayer.pack(expand=True,fill='both')
# filepath = 'videos/demovideo.mp4'
# vidplayer.load(filepath)
# vidplayer.play()
# root.mainloop() 

Instance = vlc.Instance()
player = Instance.media_player_new()
Media = Instance.media_new('videos/demovideo.mp4')
player.set_xwindow(display.winfo_id())
player.set_media(Media)
player.play()

root.mainloop()