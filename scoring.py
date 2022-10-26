import pandas as pd
import datetime
import numpy as np

timelist = []

def score(s1,s2,s3,df):
    if str(s1).lower() == "focused":
        focus = 1
    else:
        focus = 0
    if str(s2).lower() == "positive":
        emotion = 1
    else:
        emotion = 0
    if str(s3).lower() == "aligned":
        head = 1
    else:
        head = 0

    now = datetime.datetime.now()
    date = str(now.strftime("%Y-%m-%d"))
    time = str(now.strftime("%H:%M:%S:%f")[:-3])

    df_new_row = pd.DataFrame({'Date':[date],'Time':[time],'Focus':[focus],'Emotion':[emotion],'Head':[head]})
    df = pd.concat([df, df_new_row])
    
    return df

def secondScore(df):
    df2 = pd.DataFrame(columns=['Date','Time','Focus','Emotion','Head','Total'])
    row = []
    initial = 0
    startTime = datetime.datetime.strptime(df['Time'].iloc[0], "%H:%M:%S:%f")
    for i in range(len(df)):
        nextTime = datetime.datetime.strptime(df['Time'].iloc[i], "%H:%M:%S:%f")
        timedif = int((nextTime-startTime).seconds)
        if timedif == 1:
            startTime = datetime.datetime.strptime(df['Time'].iloc[i], "%H:%M:%S:%f")
            x = (df.iloc[initial:(i+1),2]).tolist() #Slicing Focus values for 1 second
            y = (df.iloc[initial:(i+1),3]).tolist() #Slicing Emotion values for 1 second
            z = (df.iloc[initial:(i+1),4]).tolist() #Slicing Head values for 1 second
            initial = i
            x2  = max(x,key = x.count)
            y2  = max(y,key = y.count)
            z2  = max(z,key = z.count)
            total = x2+y2+z2
            df2_new_row = pd.DataFrame({'Date':[df['Date'].iloc[i]],'Time':[df['Time'].iloc[i]],'Focus':[x2],'Emotion':[y2],'Head':[z2],'Total':[total]})    
            df2 = pd.concat([df2, df2_new_row],ignore_index=True)
    return df2

def videoMapping(inputdf):
    df3= inputdf
    x = len(df3)/3
    y = round(x)
    Focus = {'video1':sum((df3.iloc[0:y,2]).tolist()),'video2':sum((df3.iloc[y:(2*y),2]).tolist()),'video3':sum((df3.iloc[(2*y):(3*y),2]).tolist())}  
    Emotion = {'video1':sum((df3.iloc[0:y,3]).tolist()),'video2':sum((df3.iloc[y:(2*y),3]).tolist()),'video3':sum((df3.iloc[(2*y):(3*y),3]).tolist())}  
    Head = {'video1':sum((df3.iloc[0:y,4]).tolist()),'video2':sum((df3.iloc[y:(2*y),4]).tolist()),'video3':sum((df3.iloc[(2*y):(3*y),4]).tolist())}  
    Total = {'video1':sum((df3.iloc[0:y,5]).tolist()),'video2':sum((df3.iloc[y:(2*y),5]).tolist()),'video3':sum((df3.iloc[(2*y):(3*y),5]).tolist())}  

    filepath = 'scorelog/Videos.csv'

    outputdf = pd.read_csv(filepath)
    outputdf['Focus'] = Focus.values()
    outputdf['Emotion'] = Emotion.values()
    outputdf['Head'] = Head.values()
    outputdf['Total Score'] = Total.values()
    
    return outputdf