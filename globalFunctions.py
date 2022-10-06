import datetime
import logging
from os import stat
from sre_parse import State
import textwrap
left_eye_center = {}
state = {'open':0}



def distance (p1,p2):
    x_d = p1[0] - p2[0]
    y_d = p1[1] - p2[1]
    d = ((x_d)**2+(y_d)**2)**0.5
    return d




filedirectory='logs/'
currentDT = datetime.datetime.now()
fileName = filedirectory + currentDT.strftime("%d_%m_%Y__%H_%M_%S") + "__state.log"


# def loggingfunc(fileName):
#     loggingName.setLevel(loggingg.INFO)
#     formatter = loggingg.Formatter("%(asctime)s - %(funcName)s -%(name)s - %(levelname)s - %(message)s")
#     handler = loggingg.FileHandler(fileName,mode='w', encoding='utf-8')
#     handler.setFormatter(formatter)
#     loggingName.addHandler(handler)
#     return loggingName

# logging=loggingfunc('logging',fileName)



preState = {}

def stateMaintain(state=None):
# def stateMaintain(state=None):

    global preState,curState
    
    curState = state

    print(preState,curState)
    if preState != curState:
        if state == None:
            logFunc(f"Camera Started",'info')
        else:
            # if state == 0:
            if state['open'] == 0:
                logFunc(f"Eyes Closed. State: {state}",'info')
            # if state == 1:
            if state['open'] == 1:
                logFunc(f"Eyes Open. State: {state}","info")
            preState['open']= curState['open']

def logFunc(msg,level,log_start='inprocess'):
    # Root config
    logging.basicConfig(
        level=logging.DEBUG,
        format="{asctime} {levelname:<8} {message}",
        style='{',
        filename=fileName, 
        filemode='w')
        
    #Text Formatting
    if len(msg) > 60:
        ftext = textwrap.TextWrapper(width=70,initial_indent='',subsequent_indent=' '*33,break_long_words= False,break_on_hyphens= False).fill(text=msg)
    else:
        ftext=msg
    #Log execution
    if  log_start==True:
        logging.info('='*70)
    if level == 'debug':
        logging.debug(ftext)
    elif level == 'info':
        logging.info(ftext)
    elif level == 'warning':
        logging.warning(ftext)
    elif level == 'error':
        logging.error(ftext)
    elif level == 'critical':
        logging.critical(ftext)
    if log_start == False:
        logging.info('*'*70)


LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
LEFT_EYE_LID = [362, 385, 387, 263, 373, 380]
LEFT_IRIS = [474,475, 476, 477]

RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ] 
RIGHT_EYE_LID = [33,  160, 158, 133, 153, 144]
RIGHT_IRIS = [469, 470, 471, 472]


FACE = [103,10,332,367,152,138]
