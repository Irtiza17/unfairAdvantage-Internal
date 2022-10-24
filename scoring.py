import pandas as pd
import datetime
import numpy as np

timelist = []

def score(s1,s2,df):
    
    now = datetime.datetime.now()
    timelist.append(now)
    if str(s1).lower() == "focused":
        focus = 1
    else:
        focus = 0
    if str(s2).lower() == "positive":
        emotion = 1
    else:
        emotion = 0

    print(focus,emotion)

    now = datetime.datetime.now()

    # print ("Current date and time : ")
    # print (now.strftime("%Y-%m-%d %H:%M:%S"))


    date = str(now.strftime("%Y-%m-%d"))
    time = str(now.strftime("%H:%M:%S:%f")[:-3])

    data = {'Date':date,'Time':time,'Focus':focus,'Emotion':emotion}
    df = df.append(data, ignore_index=True)
    
    return df

def secondScore(df):
    row = []
    focscore = []
    initial = 0
    print(len(df))
    startTime = datetime.datetime.strptime(df['Time'].iloc[0], "%H:%M:%S:%f")
    for i in range(len(df)):
        nextTime = datetime.datetime.strptime(df['Time'].iloc[i], "%H:%M:%S:%f")
        timedif = nextTime-startTime
        timedif = int(timedif.seconds)
        if timedif > 1 :
            row.append(i)
            startTime = datetime.datetime.strptime(df['Time'].iloc[i], "%H:%M:%S:%f")
            xa = df['Focus'].iloc[initial:i]
            initial = i
            focscore.append()
            data = {'Date': df['Date'].iloc[i],'Time': df['Time'].iloc[i], 'Focus': xa ,'Emotion':1}
            df2 = pd.concat([df2, df.iloc[i]])
    return df2