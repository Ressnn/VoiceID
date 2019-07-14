 # -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 23:02:45 2019

@author: ASUS
"""
import tkinter 
import numpy as np
from tkinter import *
import pyaudio
import wave
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import audio_ops
import os
import glob
import keras
from keras.models import Sequential,Model
from keras.layers import Conv2D,Reshape,Activation,Dense,Dropout,BatchNormalization,Lambda,Flatten
from keras.layers import *
from keras.preprocessing.image import ImageDataGenerator
# In[]

class GUI():
    def __init__(self):
        self.GUI = tkinter.Tk()
        self.Output_Box = Text(self.GUI,height=1,width=30)
        self.Output_Box.pack()
        self.Stop=False
        self.chunk = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        self.RECORD_SECONDS = 5
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=self.FORMAT,channels=self.CHANNELS, rate=self.RATE, input=True,output=True,frames_per_buffer=self.chunk)
        self.audio = []
        self.output = "GUI/1.wav"
        self.curworkdir = r"C:\Users\ASUS\Desktop\Programming\VoiceID"
    def Update_output(self,display):
        self.Output_Box.delete(1.0,END)
        self.Output_Box.insert(INSERT,str(display))
    def OSfunc(self,x):
        return os.path.join(self.curworkdir,str(x))
    def record(self):
        print("Recording")
        self.stream.start_stream()
        counter = 0
        while counter<175:
            self.audio.append(self.stream.read(self.chunk))
            counter = counter+1
        self.stream.stop_stream()    
    def SpecWithARGS(self,inp,out):
        wav_file = tf.placeholder(tf.string)
        audio_binary = tf.read_file(wav_file)
        waveform = audio_ops.decode_wav(audio_binary, desired_channels=1)
        spectrogram = audio_ops.audio_spectrogram(waveform.audio,window_size=1024,stride=64)
        brightness = tf.placeholder(tf.float32, shape=[])
        mul = tf.multiply(spectrogram, brightness)
        min_const = tf.constant(255.)
        minimum =  tf.minimum(mul, min_const)
        expand_dims = tf.expand_dims(minimum, -1)
        resize = tf.image.resize_bilinear(expand_dims, [512, 512])
        squeeze = tf.squeeze(resize, 0)
        flip = tf.image.flip_left_right(squeeze)
        transpose = tf.image.transpose_image(flip)
        grayscale = tf.image.grayscale_to_rgb(transpose)
        cast = tf.cast(grayscale, tf.uint8)
        png = tf.image.encode_png(cast)
        with tf.Session() as sess:
            # Run the computation graph and save the png encoded image to a file
            image = sess.run(png, feed_dict={wav_file:os.path.join(self.curworkdir,str(inp)), brightness: 100})
    
            with open(os.path.join(self.curworkdir,str(out)), 'wb') as f:
                f.write(image)
    def TrainModel(self):
        print('training')
        a = os.listdir(os.path.join(self.curworkdir,'InternalStorage/Converter/Allow'))
        for i in a:
            os.remove(os.path.join(os.path.join(self.curworkdir,'InternalStorage/Converter/Allow'),i))
        b = os.listdir(os.path.join(self.curworkdir,'InternalStorage/Converter/DontAllow'))
        for i in b: 
            os.remove(os.path.join(os.path.join(self.curworkdir,'InternalStorage/Converter/DontAllow'),i))
        num = 0
        for i in os.listdir(self.OSfunc('Training_Data/Voices_To_Allow')):
            self.SpecWithARGS(self.OSfunc(os.path.join('Training_Data/Voices_To_Allow',i,)),self.OSfunc(os.path.join('InternalStorage/Converter/Allow',i))+'.png')
            num = num+1
        for i in os.listdir(self.OSfunc('Training_Data/Voices_To_Block')):
            self.SpecWithARGS(self.OSfunc(os.path.join('Training_Data/Voices_To_Block',i)),self.OSfunc(os.path.join('InternalStorage/Converter/DontAllow',i))+'.png')
            num = num+1
        image_processor = ImageDataGenerator()
        image_processor = image_processor.flow_from_directory(self.OSfunc('InternalStorage/Converter'),batch_size=num,color_mode='grayscale')
        images = image_processor[0][0]
        labels = image_processor[0][1]
        shape = images[0].shape
        self.model = Sequential()
        inp = Input(shape=(shape))
        l = Lambda(lambda x: x/255)(inp)
        C0 = Conv2D(3,(5,5),strides=(2,2))(l)
        B0 = BatchNormalization()(C0)
        C1 = Conv2D(8,(5,5),strides=(2,2),activation='relu')(B0)
        B1 = BatchNormalization()(C1)
        C2 = Conv2D(15,(5,5),strides=(2,2))(B1)
        B2 = BatchNormalization()(C2)
        C3 = Conv2D(15,(5,5),strides=(2,2),activation='relu')(B2)
        D = Dropout(.2)(C3)
        B3 = BatchNormalization()(D)
        C4 = Conv2D(30,(3,3),strides=(2,2),activation='relu')(B3)
        D1 = Dropout(.2)(C4)
        B4 = BatchNormalization()(D1)
        C5 = Conv2D(45,(2,2),strides=(1,1),activation='relu')(B4)
        Fl = Flatten()(C5)
        D3 = Dense(500,activation='relu')(Fl)
        D4 = Dropout(.3)(D3)
        B5 = BatchNormalization()(D4)
        D5 = Dense(200,activation='relu')(B5)
        B6 = BatchNormalization()(D5)
        D6 = Dropout(.4)(B6)
        D7 = Dense(100)(D6)
        B7 = BatchNormalization()(D7)
        D8 = Dense(2,activation='sigmoid')(B7)
        self.model = Model(inp,D8)
        self.model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])
        self.model.fit(images,labels,epochs=75,batch_size=10)
        
    def reset(self):
        print("Resetting")
        self.stream = self.p.open(format=self.FORMAT,channels=self.CHANNELS, rate=self.RATE, input=True,output=True,frames_per_buffer=self.chunk)
        self.audio = []
    
    def Spectrogram(self):
        wav_file = tf.placeholder(tf.string)
        audio_binary = tf.read_file(wav_file)
        waveform = audio_ops.decode_wav(audio_binary, desired_channels=1)
        spectrogram = audio_ops.audio_spectrogram(waveform.audio,window_size=1024,stride=64)
        brightness = tf.placeholder(tf.float32, shape=[])
        mul = tf.multiply(spectrogram, brightness)
        min_const = tf.constant(255.)
        minimum =  tf.minimum(mul, min_const)
        expand_dims = tf.expand_dims(minimum, -1)
        resize = tf.image.resize_bilinear(expand_dims, [512, 512])
        squeeze = tf.squeeze(resize, 0)
        flip = tf.image.flip_left_right(squeeze)
        transpose = tf.image.transpose_image(flip)
        grayscale = tf.image.grayscale_to_rgb(transpose)
        cast = tf.cast(grayscale, tf.uint8)
        png = tf.image.encode_png(cast)
        with tf.Session() as sess:
            # Run the computation graph and save the png encoded image to a file
            image = sess.run(png, feed_dict={wav_file:os.path.join(self.curworkdir,'InternalStorage\\1.wav'), brightness: 100})
    
            with open(os.path.join(self.curworkdir,'InternalStorage\\2.png'), 'wb') as f:
                f.write(image)
    
    def check(self):
        print("checking...")
        wf = wave.open(os.path.join(self.curworkdir,'InternalStorage\\1.wav'),'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(self.audio))
        self.Spectrogram()
        image_processor = keras.preprocessing.image.load_img(self.OSfunc('InternalStorage/2.png'),grayscale=True,target_size=(256,256))
        print(self.model.predict(np.array(image_processor).reshape(1,256,256,1)))
        guess = int(self.model.predict(np.array(image_processor).reshape(1,256,256,1))[0][0])
     
        
        if guess > .5:
            self.Update_output('Password accepted with probability of '+ str(guess*100)+' %')
        else:
            self.Update_output('Password Rejected')
        
    def StartGUI(self):
        self.Record = Button(self.GUI,height=1,width=30,text='Record',command=self.record)
        self.Stop = Button(self.GUI,height=1,width=30,text='TrainModel',command=self.TrainModel)
        self.Reset = Button(self.GUI,height=1,width=30,text='Reset',command=self.reset)
        self.Check = Button(self.GUI,height=1,width=30,text='Check',command=self.check)
        self.Record.pack(fill=X)
        self.Stop.pack(fill=X)
        self.Reset.pack(fill=X)
        self.Check.pack(fill=X)
        self.GUI.mainloop()
        

    

# In[]
MakeGUI = GUI()
# In[]
MakeGUI.StartGUI()
 # In[]

#%%
