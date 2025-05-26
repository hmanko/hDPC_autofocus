# -*- coding: utf-8 -*-

################################################################################
## 
################################################################################

from PyQt5 import QtWidgets, uic
import sys
import numpy as np
import pymmcore
import pandas as pd

import os.path
import matplotlib
import matplotlib.pyplot as plt
from tifffile import imread, imwrite,TiffWriter
from skimage import io
import warnings
from qtpy.QtWidgets import QApplication, QWidget, QLineEdit, QLabel,QPushButton, QProgressBar, QMessageBox, QCheckBox
from qtpy.QtGui import QFont
from qtpy import QtCore
from matplotlib.widgets  import RectangleSelector
np.seterr(divide='ignore', invalid='ignore')
from IPython import get_ipython
import glob
import time
from silx.image import sift
import silx
import math
from numpy import where
import datetime
from math import sqrt
from scipy import ndimage 
import keyboard
from scipy.optimize import curve_fit
import cv2

from pymmcore_plus import CMMCorePlus
from qtpy.QtWidgets import QApplication, QGroupBox, QHBoxLayout, QBoxLayout, QWidget,QGridLayout

from pymmcore_widgets import ExposureWidget


sys.path.append("Z:\Hanna\CODE")
from functions import *


pathMM =  "C:\Program Files\Micro-Manager-2.0_NB"   ## path to Micromanager folder on the computer
config = "PiezoArduino_Kuro.cfg" #  "ArduinoKuro.cfg" #♠"PiezoArduino_Kuro.cfg" #                 ## name of configuration file
#pymmcore.CMMCore().getAPIVersionInfo()
mmc = CMMCorePlus.instance()                             ##  initialisation of MicroManager core
mmc.setDeviceAdapterSearchPaths([pathMM])             ##  looking for device adapters
mmc.loadSystemConfiguration(os.path.join(pathMM, config))   #  Loading configuration

mmc.mda.engine.use_hardware_sequencing = True
ZStage = mmc.getFocusDevice() 
mmc.setProperty('Arduino-Switch',  'Sequence', "On")

######## the roi is defined depending on specific setup and field of view
mmc.setROI(250,250,700,700)  ##   x, y starting points, size of x and y 

global pathth
mmc.setAutoShutter(True)    
mmc.setExposure(50)

pattern = (["4","8","16","32"])  # the sequence of leds activated 
mmc.mda.engine.use_hardware_sequencing = True
mmc.loadPropertySequence("Arduino-Switch", "State", pattern)
mmc.startSequenceAcquisition(4, 0, True)
mmc.startPropertySequence("Arduino-Switch", "State")
while mmc.isSequenceRunning():
        print("test")

from multiprocessing import Process
global ref_image


mmc.setPosition(ZStage, 50)   ## the position of piezo is set in the middle
print("current position of the w drive is ",  mmc.getPosition(ZStage))

#textBrow.append("Initialization Done") 
   
def live():
    mmc.startSequenceAcquisition(20, 0, True)


def file_name_check(path):   # function to check if there is already the file with specific name
    filename, extension = os.path.splitext(path)
    counter = 2
    while os.path.exists(path):   # if the name already exists it adds the consequtive number to it
        path = filename + "_" +str(counter)+""+ extension
        counter += 1
    return path


def IDPC(sp1, sp2):   ## function to calculate differential phase contrast / gradient images
    Idpc = (sp1-sp2)/(sp1+sp2)
    return Idpc


#global ref_image   
def get_ref():
    textBrow.append("Starting reference") ## the message will be displayed in GUI
    ProgressBar.resetFormat()
    global ref_image  
    try:
        pathth
    except NameError:
        pathth = linePath.text()
    if os.path.isdir(pathth) == False:   ## if path do not exist the error message will be displayed
        textBrow.append("! Error: The path to saving folder was not defined or does not exist") 
    if mmc.isSequenceRunning() ==True:
        mmc.stopSequenceAcquisition()
    if lineStartref.text() =='':    ## checking is the tere are values entered in GUI, if not use default values
        start = -45
    else:
        start = np.float16(lineStartref.text())
    if lineStopref.text() =='':
        stop = 45
    else:
        stop = np.float16(lineStopref.text())
    if lineStepref.text() =='':
        step = 10
    else:
        step = np.float16(lineStepref.text())
    pos0 = mmc.getPosition(ZStage)
    ref_image = []
    c = 100/len(np.arange(start,stop, step))
    for i in np.arange(start,stop,step):
        mmc.setPosition(ZStage, pos0+i)
        try:
            ref_image.append(get_image(av=2))
        except RuntimeError:
            ref_image.append(get_image(av=2))
            pass
        ProgressBar.setValue(int(i*c+1))
    textBrow.append("ref acquired, saving")
    ref_image=([np.mean(np.asarray(ref_image)[:,i], axis=0) for i in range(0,4)])
    mmc.setPosition(ZStage, pos0)
    save_path = file_name_check(pathth + '\\reference.tif')
    _, tail = os.path.split(save_path)
    imwrite(save_path, np.asarray(ref_image).astype(np.float16))
    ProgressBar.setFormat("Done")
    with open(pathth + '\\METADATA_leds.txt', 'a') as file:     ## opening the Metadata file, if file do not exists it will be created automaticaly
        file.write('Ref was recorded at: %s\n Start: %s\n Stop:: %s\n Step:: %s\n' %(datetime.datetime.now(), start, stop, step))
    textBrow.append("Reference was saved")
    return np.asarray(ref_image)


def record_stack():
    ProgressBar.resetFormat()
    if mmc.isSequenceRunning() ==True:
        mmc.stopSequenceAcquisition()
    try:
        pathth
    except NameError:
        pathth = linePath.text()
    if os.path.isdir(pathth) == False:   ## if path do not exist the error message will pop up
        textBrow.append("!!! The path to saving folder was not defined or does not exist") 
    try:
        ref_image
    except NameError:
        list_of_files = glob.glob(os.path.normpath(pathth+ '\\*'))
        refpath = os.path.normpath(max([file for file in list_of_files if 'reference' in file], key=os.path.getmtime))
        ref_image = imread(refpath)
        textBrow.append("no ref")
    textBrow.append("starting Stack acquisition")
    im=[]
    contrast =[]
    stack = []
    nIm = int(nImagesStack.text())
    c = 100/nIm
    save_path = file_name_check(pathth + '\\Stack.tif')
    i=0
    mmc.loadPropertySequence("Arduino-Switch", "State", pattern)
    mmc.startSequenceAcquisition(4*nIm, 0, True)
    mmc.startPropertySequence("Arduino-Switch", "State")
    '''while mmc.isSequenceRunning():
        if mmc.getRemainingImageCount() != 0:
            image_and_MD_raw = mmc.popNextImageAndMD(fix = False)
            image_and_MD = np.asarray([image_and_MD_raw[0], image_and_MD_raw[1].json()], dtype=object)
            im.append(image_and_MD[0])
            if (len(im) % 4 == 0) == True:
                stack.append(im)
                im=[]
                ProgressBar.setValue(int(i*c+1))
                i=i+1'''
        
    with TiffWriter(save_path) as tif:
        while mmc.isSequenceRunning():
            if mmc.getRemainingImageCount() != 0:
                image_and_MD_raw = mmc.popNextImageAndMD(fix = False)
                image_and_MD = np.asarray([image_and_MD_raw[0], image_and_MD_raw[1].json()], dtype=object)
                im.append(image_and_MD[0])
                if (len(im) % nImAv == 0) == True:
                    stack = np.mean(np.array(im), axis=0)
                    tif.write(stack, contiguous=True)
                    im=[]
                    ProgressBar.setValue(int(i*c+1))
                    i=i+1                              
            if keyboard.is_pressed('q'):  # Check if 'q' is pressed
                textBrow.append("Interrupted by user")
                break   
            if keyboard.is_pressed('s'):  # Check if 'q' is pressed
                textBrow.append("Interrupted by user")
                break 
    #textBrow.append("Saving") 
    #imwrite(file_name_check(pathth + '\\Stack_orig.tif'), np.asarray(stack).astype(np.float32))  
    with open(pathth + '\\METADATA_leds.txt', 'a') as file:     ## opening the Metadata file, if file do not exists it will be created automaticaly
        file.write('Stack was recorded at: %s\n N of images: %s\n' %(datetime.datetime.now(), nIm))
    textBrow.append("Done")             
    
    
def get_image(av=1):   ## halping function, acquires 4 images with 4 leds
    im=[]
    mmc.loadPropertySequence("Arduino-Switch", "State", pattern)
    mmc.startSequenceAcquisition(4*av, 0, True)
    mmc.startPropertySequence("Arduino-Switch", "State")
    while mmc.isSequenceRunning():
        if mmc.getRemainingImageCount() != 0:
            image_and_MD_raw = mmc.popNextImageAndMD(fix = False)
            image_and_MD = np.asarray([image_and_MD_raw[0], image_and_MD_raw[1].json()], dtype=object)
            im.append(image_and_MD[0])
    return im


def record_z_stack():
    mmc.setProperty('Arduino-Switch',  'Sequence', "On")
    #mmc.mda.engine.use_hardware_sequencing = True
    mmc.loadPropertySequence("Arduino-Switch", "State", pattern)
    mmc.startPropertySequence("Arduino-Switch", "State")
    ProgressBar.resetFormat()
    if mmc.isSequenceRunning() ==True:
        mmc.stopSequenceAcquisition()
    startTime = time.time()
    cv2.namedWindow('Live')
    textBrow.append("Starting zStack acquisition")
    try:
        pathth
    except NameError:
        pathth = linePath.text()
    if os.path.isdir(pathth) == False:   ## if path do not exist the error message will pop up
        textBrow.append("!!! The path to saving folder was not defined or does not exist") 
    if lineStartzStack.text() =='':    ## checking is the tere are values entered in GUI, if not use default values
        start = -1.5
    else:
        start = np.float16(lineStartzStack.text())
    if lineStopzStack.text() =='':
        stop = 1.5
    else:
        stop = np.float16(lineStopzStack.text())
    if lineStepzStack.text() =='':
        step = 0.05
    else:
        step = np.float16(lineStepzStack.text())
    list_of_files = glob.glob(os.path.normpath(pathth+ '\\*'))
    refpath = os.path.normpath(max([file for file in list_of_files if 'reference' in file], key=os.path.getmtime))
    ref_image = imread(refpath)
    zStack = []
    drivePos=[]
    pos0 = mmc.getPosition(ZStage)
    c = 100/len(np.arange(start, stop, step))
    
    '''i_list = []
    i=start
    while i < stop:
        i_list.append(i)
        if (i>-0.1) and (i <0.1):
            print('step = 0.01:' , i)
            i=i+0.01
        elif (i<-1.)or(i>1):
            print('step = 0.1', i)
            i=i+0.2
        elif (i<-0.5)or(i>0.5):
            print('step = 0.1', i)
            i=i+0.1
        else:
            print(i)
            i=i+step'''
    for i in np.arange(start, stop, step):
        mmc.setPosition(ZStage, pos0+i)
        try:
            zStack.append(get_image(av=2))
        except RuntimeError:
            zStack.append(get_image(av=2))
            pass
        textBrow.append('z drive position = '+str(mmc.getPosition()))
        drivePos.append(mmc.getPosition(ZStage))
        ProgressBar.setValue(int(i*c+1))
        if keyboard.is_pressed('q'):  # Check if 'q' is pressed
            textBrow.append("Interrupted by user")
            break   
    mmc.setPosition(ZStage, pos0)    
    textBrow.append("Sacindg the data")
    save_path = file_name_check(pathth + '\\z_ledStack_dif.tif')
    ppp, tail = os.path.split(save_path)
    imwrite(file_name_check(ppp+'\\z_ledStack_orig.tif'), np.asarray(zStack).astype(np.float32))
    np.savetxt(file_name_check(pathth + '\\zStack_posRead.txt'), drivePos)
    textBrow.append("Done")
    print("it took : ", time.time() - startTime)
    with open(pathth + '\\METADATA_leds.txt', 'a') as file:     ## opening the Metadata file, if file do not exists it will be created automaticaly
        file.write('zStack was recorded at: %s\n Start: %s\n Stop:: %s\n Step:: %s\n' %(datetime.datetime.now(), start, stop, step))
    
def acq_autofocus():    ## The function to perform acquisition with active autofocusing
    mmc.setProperty('Arduino-Switch',  'Sequence', "On")
    #mmc.mda.engine.use_hardware_sequencing = True
    mmc.loadPropertySequence("Arduino-Switch", "State", pattern)
    mmc.startPropertySequence("Arduino-Switch", "State")
    if mmc.isSequenceRunning() ==True:
        mmc.stopSequenceAcquisition()
    ProgressBar.resetFormat()
    startTime = time.time()
    textBrow.append("Starting acquisition with active Autofocusing")
    try:
        pathth
    except NameError:
        pathth = linePath.text()
    if os.path.isdir(pathth) == False:   ## if path do not exist the error message will pop up
        textBrow.append("!!! The path to saving folder was not defined or does not exist") 
    list_of_files = glob.glob(os.path.normpath(pathth+ '\\*'))
    refpath = os.path.normpath(max([file for file in list_of_files if 'reference' in file], key=os.path.getmtime))
    ref_image = imread(refpath)  
    pospath = os.path.normpath(max([file for file in list_of_files if 'posRead' in file], key=os.path.getmtime))
    positionList = np.loadtxt(pospath)
    zStackpath = os.path.normpath(max([file for file in list_of_files if 'z_ledStack' in file], key=os.path.getmtime))
    _, tail = os.path.split(zStackpath)
    z_Stack =  imread(zStackpath)
    Idpc1 = IDPC(np.asarray(z_Stack)[:,3]/ref_image[3], np.asarray(z_Stack)[:,0]/ref_image[0]) 
    Idpc2 = IDPC(np.asarray(z_Stack)[:,2]/ref_image[2], np.asarray(z_Stack)[:,1]/ref_image[1]) 
    z_Stack=Idpc2-Idpc1
    nIm = int(lineImAutofocus.text())
    degree = 10
    frequency = int(lineFreqAutofocus.text())
    micPos=[]
    positionList = positionList-positionList.min()
    x = positionList-positionList[int(len(positionList)/2)]
    xx = np.arange(x.min(), x.max(), 0.001)
    frame = int(len(z_Stack)/2)
    sift_ocl = sift.SiftPlan(template=z_Stack[frame], devicetype="GPU")
    keypoints = sift_ocl(z_Stack[frame])
    mp = sift.MatchPlan()
    
    plt.ion()   ## to enable changing the graph in real time
    xg = np.linspace(0, nIm, nIm)  # we need to define range before for x axis
    yg = np.arange(-1.5, 1.5, 3/nIm) #  and for y-axis
    ###
    fig = plt.figure()  
    ax = fig.add_subplot(111)
    line1, = ax.plot(xg, yg, 'b+:')
    plt.xlabel("Refocusing step")
    plt.ylabel("Drift, μm")
    plt.title("Updating plot...") 
    yyg = np.zeros(nIm)
    yyg[:]= np.nan
    k=0
    
    plt.ion()
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    line2, = ax2.plot(x, np.linspace(0, 0.0005, len(z_Stack)), 'b+:')
    line3, = ax2.plot(x, np.linspace(0, 0.0005, len(z_Stack)), 'r')
    plt.xlabel("Refocusing step")
    plt.ylabel("CCC, μm")
    plt.title("Updating plot...") 
    ProgressBar.setValue(0)
    c = 100/nIm
    imag = []
    drft=[]
    difI=[]
    micPos=[]
    for i in range(0,nIm):
        if keyboard.is_pressed('q'):  # Check if 'q' is pressed
            textBrow.append("Interrupted by user")
            break 
        if keyboard.is_pressed('s'):  # Check if 'q' is pressed
            textBrow.append("Interrupted by user, the data will be saved")
            np.savetxt(file_name_check(pathth+'/detectedDrift_interupt'+tail[-5]+'.txt'), pd.concat([pd.DataFrame(drft), pd.DataFrame(micPos)], axis=1))            
            imwrite(file_name_check(pathth + '\\Stack_autofocus_ccc_interupt'+tail[-5]+'.tif'), np.asarray(difI).astype(np.float16))
            textBrow.append("Done, interupted")
            break 
        try:
            image = get_image(av=2)
        except RuntimeError:
            image = get_image(av=2)
            pass
        Idpc1 = IDPC(np.asarray(image)[3]/ref_image[3], np.asarray(image)[0]/ref_image[0]) 
        Idpc2 = IDPC(np.asarray(image)[2]/ref_image[2], np.asarray(image)[1]/ref_image[1])     
        difI.append(Idpc2-Idpc1)
        if (len(difI) % frequency == 0) == True:
            pos = mmc.getPosition(ZStage)
            corrcoef = []
            try: 
                im_keypoints = sift_ocl(np.mean(np.asarray(difI)[i-1:i],axis=0).astype(np.float32))
                match = mp(keypoints, im_keypoints)
                sa = sift.LinearAlign(z_Stack[frame], devicetype="GPU")   
                im = sa.align(np.mean(np.asarray(difI)[i-1:i], axis=0).astype(np.float32),  shift_only=True)
            except:
                textBrow.append("Something went wrong with xy drift corretion, saving the data...")
                np.savetxt(file_name_check(pathth+'/detectedDrift_int_err.txt'), pd.concat([pd.DataFrame(drft), pd.DataFrame(micPos)], axis=1))
                ProgressBar.setFormat("Done")
                save_path = file_name_check(pathth + '\\Stack_autofocus_err.tif')
                _, tail = os.path.split(save_path)
                imwrite(save_path, np.asarray(difI).astype(np.float16))
            for f in range(0, len(z_Stack)):
                try:
                    corrcoef.append(ncc(im[100:600, 100:600], z_Stack[f,100:600,100:600]))
                except:
                    np.savetxt(file_name_check(pathth+'/detectedDrift.txt'), pd.concat([pd.DataFrame(drft), pd.DataFrame(micPos)], axis=1))
                    ProgressBar.setFormat("Done, data saved after xy-drfit correction didn't work")
                    imwrite(file_name_check(pathth + '\\Stack_autofocus_ccc_err.tif'), np.asarray(difI).astype(np.float16))
                    with open(pathth + '\\METADATA.txt', 'a') as file:
                        file.write('Aqqusition with autofocus recorded FAIL correlation at %s\n N of images %s \n frequency:%s\n Exposure: %s\n  degree: %s\n ' %(datetime.datetime.now(), nIm, frequency, exposure, degree))
                    print('xy correction failed, data was saved')
                    break
            y = np.array(corrcoef)
            model2 = np.poly1d(np.polyfit(x, y, degree))
            drft.append(xx[where(model2(xx) == model2(xx).max())])
 
            yyg[i] = drft[-1]
            line1.set_ydata(yyg)
            k = k+1
            fig.canvas.draw()   
            fig.canvas.flush_events()
            
            ax2.set_ylim(y.min()-y.min()*0.1, y.max()+y.max()*0.1)
            line2.set_ydata(y)
            line3.set_ydata(model2(x))
            fig2.canvas.draw()   
            fig2.canvas.flush_events()
            micPos.append(mmc.getPosition(ZStage))
            if keyboard.is_pressed('s'):  # Check if 'q' is pressed
                textBrow.append("Interrupted by user, the data will be saved")
                np.savetxt(file_name_check(pathth+'/detectedDrift_interupt'+tail[-5]+'.txt'), pd.concat([pd.DataFrame(drft), pd.DataFrame(micPos)], axis=1))
                imwrite(file_name_check(pathth + '\\Stack_autofocus_ccc_interupt'+tail[-5]+'.tif'), np.asarray(difI).astype(np.float16))
                textBrow.append("Done, interupted")
                break 
            textBrow.append('newvalue'+str(drft[-1]))
            if (np.abs(drft[-1]) < 0.010):
                textBrow.append('drift was smaller then 10 nm, no correction is happening')
                continue 
            mmc.setPosition(ZStage, pos-drft[-1][0])
            mmc.waitForDevice(ZStage)
        ProgressBar.setValue(int(i*c+1))
        i=i+1
        if keyboard.is_pressed('q'):  # Check if 'q' is pressed
            textBrow.append("Interrupted by user")
            break 
    plt.figure()
    plt.plot(micPos)    
    plt.title("Positions read from z drive")
    plt.show()    
    textBrow.append("Saving the data ...")    
    np.savetxt(file_name_check(pathth+'/detectedDrift.txt'), pd.concat([pd.DataFrame(drft), pd.DataFrame(micPos)], axis=1))
    imwrite(file_name_check(pathth + '\\Stack_autofocus_ccc.tif'), np.asarray(difI).astype(np.float16))
    textBrow.append("Done")
    with open(pathth + '\\METADATA_leds.txt', 'a') as file:     ## opening the Metadata file, if file do not exists it will be created automaticaly
        file.write('autofocusStack was recorded at: %s\n degree: %s\n frequency: %s\n' %(datetime.datetime.now(), degree, frequency))

def start_live():
    im=[]
    #mmc.setExposure(50)
    cv2.namedWindow('Live')
    mmc.startContinuousSequenceAcquisition()
    mmc.startPropertySequence("Arduino-Switch", "State")
    while True:
        if mmc.getRemainingImageCount() > 0:
            rgb32 = mmc.popNextImage()
            im.append(rgb32)
            if (len(im) % 4 == 0) == True:
                Idpc1 = np.asarray(im)[0]- np.asarray(im)[3]
                Idpc2 = np.asarray(im)[2]- np.asarray(im)[1]
                dif=(Idpc2-Idpc1).astype(dtype=np.uint16)
                dif = dif/(dif.max()/250)
                cv2.imshow('Live', cv2.resize(dif.astype(dtype=np.uint8), (700, 700)) )
                im=[]
        if cv2.waitKey(1) & 0xFF == ord('q'):
            mmc.stopSequenceAcquisition()  
            break
    cv2.destroyAllWindows()

def start_live_all():   ## this is the fucntion to start the live view while using as illumination all leds... 
    im=[]                #...at the same time or one by one
    mmc.stopPropertySequence("Arduino-Switch", "State")  ## to enable activation of leds without its sequence
    mmc.setProperty('Arduino-Switch',  'Sequence', "Off")  ## 
    #mmc.setExposure(50)
    cv2.namedWindow('Live')
    if comboBox.currentText() == 'all leds':   # it checks which option was choosed in the GUI
        print('all leds are activated')
        mmc.setProperty("Arduino-Switch",  "State", '60')    # to activae all leds 
    if comboBox.currentText() == 'led 1':
        print('led 1 is activated')
        mmc.setProperty("Arduino-Switch",  "State", '4')
    if comboBox.currentText() == 'led 2':
        print('led 2 is activated')
        mmc.setProperty("Arduino-Switch",  "State", '8')
    if comboBox.currentText() == 'led 3':
        print('led 3 is activated')
        mmc.setProperty("Arduino-Switch",  "State", '16')
    if comboBox.currentText() == 'led 4':
        print('led 4 is activated')
        mmc.setProperty("Arduino-Switch",  "State", '32')
    mmc.startContinuousSequenceAcquisition()
    mmc.setProperty('Arduino-Shutter',  'OnOff', '1')  # opening the shuter
    while True:   ##  continuous acquisition with visualisation 
        if mmc.getRemainingImageCount() > 0:
            rgb32 = mmc.popNextImage()
            im = rgb32.astype(dtype=np.uint16)
            im = im/(im.max()/250)
            cv2.imshow('Live', cv2.resize(im.astype(dtype=np.uint8), (700, 700)) )
            im=[]
        if cv2.waitKey(1) & 0xFF == ord('q'):
            mmc.stopSequenceAcquisition()  
            break
    cv2.destroyAllWindows()
    mmc.setProperty('Arduino-Switch',  'Sequence', "On")
    #mmc.mda.engine.use_hardware_sequencing = True
    mmc.loadPropertySequence("Arduino-Switch", "State", pattern)
    mmc.startPropertySequence("Arduino-Switch", "State")


def setZpos():   # function to set new position to the z-drive
    npos = np.float16(setzPos.text())
    pos0 = mmc.getPosition(ZStage)
    mmc.setPosition(ZStage, pos0+npos)
    textBrow.append("The position of z-drive was changed by"+str(npos)) 
    


########################################################################################
########################################################################   Gui window
########################################################################################   
exposure = 50

class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()
        ui = uic.loadUi('Z:/Hanna/CODE/LedControl/ledControl2.ui', self)
        global lineStartref,lineStopref,lineStepref,lineStartzStack,lineStopzStack,lineStepzStack,linePath, nImagesStack,setzPos
        global ProgressBar,acqAFline, freqAFline,lineImAutofocus,lineFreqAutofocus,comboBox,textBrow
        
        lineStartref = ui.lineStartref
        lineStopref = ui.lineStopref
        lineStepref = ui.lineStepref 
        lineStartzStack = ui.lineStartzStack
        lineStopzStack = ui.lineStopzStack
        lineStepzStack = ui.lineStepzStack 
        linePath = ui.linePath
        setzPos = ui.setzPos
        nImagesStack = ui.nImagesStack
        
        textBrow = ui.MotherFuckingText
        
        comboBox = ui.comboBox
        
        lineImAutofocus = ui.lineImAutofocus
        lineFreqAutofocus = ui.lineFreqAutofocus
        
        autofocusButton = ui.autofocusButton
        autofocusButton.clicked.connect(acq_autofocus)
   
        ProgressBar = ui.progressBar
               
        #self.startlive = self.findChild(QtWidgets.QPushButton, 'StartLive') # Find the button

        #self.startlive.clicked.connect(start_live) # Remember to pass the definition/method, not the return value!
            
        self.startlive = self.findChild(QtWidgets.QPushButton, 'StartLive_all')
        self.startlive.clicked.connect(start_live_all)
        
        self.recordStack = self.findChild(QtWidgets.QPushButton, 'recordStack')
        self.recordStack.clicked.connect(record_stack)
        
        self.record_zStack = self.findChild(QtWidgets.QPushButton, 'zStackRec')
        self.record_zStack.clicked.connect(record_z_stack)
        
        self.getref = self.findChild(QtWidgets.QPushButton, 'GetRef')
        self.getref.clicked.connect(get_ref)
        
        self.setZ = self.findChild(QtWidgets.QPushButton, 'setzPosButton') # Find the button
        self.setZ.clicked.connect(setZpos)


        expButton = ExposureWidget()
        
        self.layot_1 = self.findChild(QtWidgets.QHBoxLayout, 'explayout') # Find the button
        self.layot_1.addWidget(expButton) # Remember to pass the definition/method, not the return value!

        self.show()      
app = QtWidgets.QApplication(sys.argv)
window = Ui()
window.show()
app.exec_()


