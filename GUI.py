# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 20:57:51 2017

@author: Esmeralda Chang
"""


from tkinter import *
from ScrolledText import ScrolledText

import numpy as np
import cv2
import thread
import time
from keras import backend as K
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.models import load_model
from keras.models import Sequential, model_from_json
from keras.layers import *
from PIL import ImageTk, Image
import os
import threading
import requests


W  = '\033[0m'  # white (normal)
R  = '\033[31m' # red
G  = '\033[32m' # green
O  = '\033[33m' # orange
B  = '\033[34m' # blue
P  = '\033[35m' # purple

class RequestThread (threading.Thread):
    maxRetries = 20
    def __init__(self, imagepath, ip, port, thread_lock):
        threading.Thread.__init__(self)
        self.imagepath = imagepath
        self.thread_lock = thread_lock
        self.ip = ip
        self.port = port
        
    def run(self):
        url = 'http://{0}:{1}/config/info.cgi'.format(self.ip,self.port)
        files = {'media': open(self.imagepath, 'rb')}
        requests.post(url, files=files)
           
           
class PredictThread (threading.Thread):
    maxRetries = 20
    def __init__(self, GUI, cnnmodel, rgbs, frames, thread_lock):
        threading.Thread.__init__(self)
        self.rgbs = rgbs
        self.frames = frames
        self.thread_lock = thread_lock
        self.cnnmodel = cnnmodel;
        self.GUI = GUI

    def cnn_frame(self, cnnmodel,rgbs):
        rgbs = rgbs.reshape(3,3,48,64,18)
        Flatten = cnnmodel.predict(rgbs)
        return Flatten
    
    def run(self):
        t1= time.clock()
        rgbs = self.cnn_frame(self.cnnmodel,self.rgbs)
        print('it took {0} cnn seconds'.format(time.clock() - t1))
        self.GUI.printmsg('it took {0} cnn seconds'.format(time.clock() - t1))
        rgbs = np.array(rgbs)
        rgbs = rgbs.reshape(1, 3, 512)
        rgbs = rgbs.astype('float64')
        t1= time.clock()
        p = model.predict(rgbs)
        print('it took {0} predict seconds'.format(time.clock() - t1))
        self.GUI.printmsg('it took {0} predict seconds'.format(time.clock() - t1))
        print('predict_{0}'.format(p))
        self.GUI.printmsg('predict_{0}'.format(p))
        if(p[:,1] > 0.5):
            print(R+"fall down"+W)
            self.GUI.printmsg("fall down")
            img = '\\fall{0}.jpg'.format(time.clock())
            cv2.imwrite(img,self.frames[10])
            request_thread = RequestThread(img,'127.0.0.1','1688',self.thread_lock)
            request_thread.start() 
        else:
            print('normal')
            self.GUI.printmsg('normal')

class WritevideoThread(threading.Thread):
    maxRetries = 20
    def __init__(self, thread_lock):
        threading.Thread.__init__(self)
        self.thread_lock = thread_lock
        self.thread_start = True
        
    def run(self):
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        h,w,d = frame.shape
        frame = cv2.resize(frame,(w/2, h/2), interpolation = cv2.INTER_AREA)
        out_avi = '\\webcam.avi'
        print('out_avi'+out_avi)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(out_avi, fourcc, 25.0, (w/2,h/2))  

        while(self.thread_start == True):
            # Capture frame-by-frame
            ret, frame = cap.read()
            frame = cv2.resize(frame,(w/2, h/2), interpolation = cv2.INTER_AREA)
            out.write(frame)
            
        cap.release()
        out.release()
        cv2.destroyAllWindows()

class StreamingThread(threading.Thread):
    maxRetries = 20
    def __init__(self,GUI, video, thread_lock):
        threading.Thread.__init__(self)
        self.thread_lock = thread_lock
        self.GUI = GUI
        self.thread_start = True
        self.video = video
        
    def run(self):
        vid = self.video  
        print(vid)
        use_webcam = False
        pointer = 1
        index = 0
        min = 3
        resize_h = 150
        resize_w = 200
        
        if(use_webcam == True):
            webcam = cv2.VideoCapture(0)
            ret, frame = webcam.read()
        else:
            webcam = cv2.VideoCapture(vid)
            ret, frame = webcam.read(pointer+index)
            
        frame = cv2.resize(frame,(resize_w, resize_h), interpolation = cv2.INTER_AREA)                                                             
        prvs = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame)
        hsv[...,1] = 255
        index = index +1
        rgbs = []
        frames = []

        while(self.thread_start == True):
            if(use_webcam == True):
                ret, frame = webcam.read()
            else:
                ret,frame = webcam.read(pointer+index)
            if(ret == False):
               break
            
            if(ret == True):
                resizeframe = cv2.resize(frame,(320, 240), interpolation = cv2.INTER_AREA)
                cv2.imshow('frame',resizeframe)
                cv2.waitKey(1)
                frame = cv2.resize(frame,(resize_w, resize_h), interpolation = cv2.INTER_AREA)
                if(index == 1):
                    t1= time.clock()
                    
                frames.append(frame)
                next = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                fx, fy = flow[:,:,0], flow[:,:,1]
                ang = np.arctan2(fy, fx) + np.pi
                v = np.sqrt(fx*fx+fy*fy)
                hsv[...,0] = ang*(180/np.pi/2)
                hsv[...,2] = np.minimum(v*min, 255)
                rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                
            
                rgb=cv2.resize(rgb,(64,48),interpolation=cv2.INTER_AREA)
                rgbs.append(rgb)
                prvs = next
            
                index = index +1
            if(index == 55):
                self.GUI.printmsg('it took {0} optical flow seconds'.format(time.clock() - t1))
                print('it took {0} optical flow seconds'.format(time.clock() - t1))
                rgbs = np.array(rgbs)
             
                thread_lock = threading.Lock()
                predic_thread = PredictThread(self.GUI, cnnmodel, rgbs, frames, thread_lock)
                predic_thread.start()
    
                rgbs = []
                frames = []
                pointer = pointer + 55
                index = 0
                
                if(use_webcam == True):
                    ret, frame = webcam.read()
                else:
                    ret,frame = webcam.read(pointer+index)
                
                frame = cv2.resize(frame,(resize_w, resize_h), interpolation = cv2.INTER_AREA)
                prvs = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                hsv = np.zeros_like(frame)
                hsv[...,1] = 255
                index = index +1 
                
        webcam.release()
        cv2.destroyAllWindows()
            
            
        
class GUIDemo(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.grid()
        self.createWidgets()
        self.thread_lock = threading.Lock()
   
    def cnn_frame(self, cnnmodel,rgbs):
            rgbs = rgbs.reshape(3,3,48,64,18)
            Flatten = cnnmodel.predict(rgbs)
            return Flatten
    
    def preparemodel(self,model):
        vid = '\\preparemodel\\chute(1).avi'
        cap = cv2.VideoCapture(vid)
        img_rows,img_cols=48,64
        rgbs = []
        while(1):
            ret, frame = cap.read()
            if frame is None:
                break;
            frame=cv2.resize(frame,(img_cols,img_rows),interpolation=cv2.INTER_AREA)
            rgbs.append(frame)
    
        rgbs = np.array(rgbs)
        frames = self.cnn_frame(model,rgbs)
        cap.release()
        cv2.destroyAllWindows()
        return frames
     
    def loadmodel(self):
        global model
        global cnnmodel
        dir = '\\Model\\'
        config = open('{0}\\falldown.json'.format(dir), "rb").read()
        model = model_from_json(config)
        model.load_weights('{0}\\falldown-weight.h5'.format(dir))
        cnnmodel = load_model('{0}\\cnn.h5'.format(dir))
        self.preparemodel(cnnmodel)
        K.set_image_dim_ordering('th')
        
    def printmsg(self,msg):
        self.inputField.insert(END, msg+'\n')
        self.inputField.see(END)
        
    def stop(self):
        self.stream_thread.thread_start = False
        self.write_thread.thread_start = False
        
    def startdetection(self):
        self.printmsg('loadmodel')
        print("loadmodel")
        self.loadmodel()  
        
        if(self.checkvar.get() == 1):
            self.write_thread = WritevideoThread(self.thread_lock)        
            self.write_thread.start()
        self.stream_thread = StreamingThread(self,self.videoFile.get(),self.thread_lock)
        self.stream_thread.start()    
        
        
    def createWidgets(self):
        self.space = Label(self)
        self.space.grid(row=1, column=0)
        
        """Turn on webcam"""
        self.checkvar = IntVar()
        self.check = Checkbutton(self, variable=self.checkvar)
        self.check["text"] = "Turn on webcam"
        self.check.grid(row=2, column=4) 
                     
        """start detection"""
        self.new = Button(self,command=self.startdetection)
        self.new["text"] = "  Start  "
        self.new.grid(row=2, column=2, columnspan=1)
        
        """stop detection"""
        self.new = Button(self,command=self.stop)
        self.new["text"] = "  Stop  "
        self.new.grid(row=2, column=3, columnspan=1)
        self.space = Label(self)
        self.space.grid(row=3, column=0)
        
        """video file"""
        self.videoText = Label(self)
        self.videoText["text"] = "video:"
        self.videoText.grid(row=4, column=0)
        self.videoFile = Entry(self)
        self.videoFile["width"] = 50
        self.videoFile.grid(row=4, column=1, columnspan=6)
        self.videoFile.insert(0,'D:\\MSLab\\thesis\\fall down data\\extractframe\\webcam.avi')     
        
        self.space = Label(self)
        self.space.grid(row=5, column=0)
        
        """detection result"""
        self.inputText = Label(self)
        self.inputText["text"] = "  Result:  "
        self.inputText.grid(row=6, column=0)
        self.inputField = ScrolledText(self)
        self.inputField["width"] = 50
        self.inputField.grid(row=6, column=1, columnspan=6) 
        
        self.space = Label(self)
        self.space.grid(row=7, column=0)
        
        
 
if __name__ == '__main__':
    root = Tk()
    root.resizable(width=False, height=False)
    root.minsize(width=450, height=450)
    
    app = GUIDemo(master=root)
    app.mainloop()
    
    