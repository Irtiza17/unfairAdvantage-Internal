import pandas as pd
import datetime
import numpy as np

timelist = []


class Scoring():

    def __init__(self):
        self.filepath = 'scorelog/SingleFrameData.csv'  #csv file for score data of every frame.
        self.filepath2 = 'scorelog/PerSecondData.csv' # csv file for score data of every second.
        self.filepath3 = 'scorelog/Report.csv' # csv file for videos information and score mapping against videos.
        self.df = pd.DataFrame(columns=['Date','Time','Focus','Emotion','Head','Pointing','Waving','Hand Movement'])

    def singleFrameDate(self,s1,s2,s3,s4,s5,s6,timeVal):
    
        # Calculating score based on Focus, Emotion and Head Values
        if str(s1).lower() == "focused":
            focus = 1
        else:
            focus = 0
        if str(s2).lower() == "positive":
            emotion = 1
        else:
            emotion = 0
        if str(s3).lower() == "center":
            head = 1
        else:
            head = 0
        if str(s4).lower() == "pointing":
            hand = 1
        else:
            hand = 0
        if str(s5).lower() == "waving":
            waving = 1
        else:
            waving = 0
        if str(s6).lower() == "moving" or str(s5).lower() == "waving":
            moving = 1
        else:
            moving = 0

        date = str(timeVal.strftime("%Y-%m-%d"))
        time = str(timeVal.strftime("%H:%M:%S:%f")[:-3])

        df_new_row = pd.DataFrame({'Date':[date],'Time':[time],'Focus':[focus],'Emotion':[emotion],'Head':[head],'Pointing':[hand],'Waving':[waving],'Hand Movement':[moving]})
        self.df = pd.concat([self.df, df_new_row])

    def secondScore(self):
        self.df.to_csv(self.filepath,index=False)

        df2 = pd.DataFrame(columns=['Date','Time','Focus','Emotion','Head','Pointing','Waving','Hand Movement','Total'])
        initial = 0
        startTime = datetime.datetime.strptime(self.df['Time'].iloc[0], "%H:%M:%S:%f")
        for i in range(len(self.df)):
            nextTime = datetime.datetime.strptime(self.df['Time'].iloc[i], "%H:%M:%S:%f")
            timedif = int((nextTime-startTime).seconds)
            if timedif == 1:
                startTime = datetime.datetime.strptime(self.df['Time'].iloc[i], "%H:%M:%S:%f")
                x = (self.df.iloc[initial:(i+1),2]).tolist() #Slicing Focus values for 1 second and converting to list
                y = (self.df.iloc[initial:(i+1),3]).tolist() #Slicing Emotion values for 1 second and converting to list
                z = (self.df.iloc[initial:(i+1),4]).tolist() #Slicing Head values for 1 second and converting to list
                w = (self.df.iloc[initial:(i+1),5]).tolist() #Slicing Head values for 1 second and converting to list
                u = (self.df.iloc[initial:(i+1),6]).tolist() #Slicing Head values for 1 second and converting to list
                v = (self.df.iloc[initial:(i+1),7]).tolist() #Slicing Head values for 1 second and converting to list
                initial = i
                x2  = max(x,key = x.count) #getting the most frequent score value for Focus
                y2  = max(y,key = y.count) #getting the most frequent score value for Emotion
                z2  = max(z,key = z.count) #getting the most frequent score value for Head
                w2  = max(w,key = w.count) #getting the most frequent score value for Head
                u2  = max(u,key = u.count) #getting the most frequent score value for Head
                v2  = max(v,key = v.count) #getting the most frequent score value for Head
                total = x2+y2+z2+w2+u2+v2 #Total score 

                df2_new_row = pd.DataFrame({'Date':[self.df['Date'].iloc[i]],'Time':[self.df['Time'].iloc[i]],'Focus':[x2],'Emotion':[y2],'Head':[z2],'Pointing':[w2],'Waving':[u2],'Hand Movement':[v2],'Total':[total]})    
                df2 = pd.concat([df2, df2_new_row],ignore_index=True) #Adding the values to the dataframe
        df2.to_csv(self.filepath2,index=False)        
        return df2 

    def reportGeneration(self):
        #Function to map the scores corresponding the videos and their intelligence tags
        df3 = self.secondScore()
        filepath = 'scorelog/Videos.csv'
        outputdf = pd.read_csv(filepath)
        lendf = len(outputdf)
        x = len(df3)/lendf
        print(lendf)
        y = round(x)
        Focus={}
        Emotion={}
        Head={}
        Pointing={}
        Waving={}
        Moving={}
        Total={}
        for i in range(lendf):
            Focus[f'video{i+1}'] = sum((df3.iloc[i*y:(i+1)*y,2]).tolist())  
            Emotion[f'video{i+1}'] = sum((df3.iloc[i*y:(i+1)*y,3]).tolist())  
            Head[f'video{i+1}'] = sum((df3.iloc[i*y:(i+1)*y,4]).tolist())  
            Pointing[f'video{i+1}'] = sum((df3.iloc[i*y:(i+1)*y,5]).tolist())  
            Waving[f'video{i+1}'] = sum((df3.iloc[i*y:(i+1)*y,6]).tolist())  
            Moving[f'video{i+1}'] = sum((df3.iloc[i*y:(i+1)*y,7]).tolist())  
            Total[f'video{i+1}'] = sum((df3.iloc[i*y:(i+1)*y,8]).tolist())  
        
        
        outputdf['Focus'] = Focus.values()
        outputdf['Emotion'] = Emotion.values()
        outputdf['Head'] = Head.values()
        outputdf['Pointing'] = Pointing.values()
        outputdf['Waving'] = Waving.values()
        outputdf['Hand Movement'] = Moving.values()
        outputdf['Total Score'] = Total.values()

        outputdf.to_csv(self.filepath3,index=False)
