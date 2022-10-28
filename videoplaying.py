# importing vlc module


import vlc
 
# importing time module
import time

def videoplayer(): 
# creating vlc media player object
    media_player = vlc.MediaPlayer()
    
    # media object
    media = vlc.Media("videos/video1.mp4")
    
    # setting media to the media player
    media_player.set_media(media)
    
    
    # start playing video
    media_player.play()
    
    # wait so the video can be played for 5 seconds
    # irrespective for length of video
    # time.sleep()media = vlc.MediaPlayer("video.mp4")
    playing = set([1,2,3,4])
    play = True
    while play:
        time.sleep(0.5)
        state = media_player.get_state()
        if state in playing:
            continue
        else:
            play = False