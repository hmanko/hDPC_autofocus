# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 15:54:00 2024

@author: hmanko
"""

 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 18:17:01 2023

@author: hannamanko

"""

"""
The gui created in order to control Nikon Ti microscope with Phasics camera. This version of GUI was created using QtDesigner 
(can be started using 'qt5-tools designer' from the command promt). 



The soft was developed to be started on windows system, if you are going to use it on any 
other system it will be required to change the way how the path for saving data is defined 
in all the functions.
Ex: for windows before the file name separated by '\\' need to be written, as it was done in this code 


The lines where changes could be required are marked as ############ ***
                                                        ####   ***   !!!

"""
###   importing all the required libraries 
import pymmcore_plus
import numpy as np
import cv2
import os.path
from pymmcore_plus import CMMCorePlus
import useq
from useq import MDAEvent, MDASequence
from pymmcore_plus.mda import MDAEngine
import matplotlib.pyplot as plt
from IPython import get_ipython
from numpy import median


import pymmcore
import pandas as pd
import numpy as np
import napari
import os.path
import matplotlib
#matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from tifffile import imread, imwrite
from skimage import io
import sys
from matplotlib.widgets import Button
import warnings
from qtpy.QtWidgets import QApplication, QWidget, QLineEdit, QLabel,QPushButton, QProgressBar, QMessageBox, QCheckBox
from qtpy.QtGui import QFont
from qtpy import QtCore
from matplotlib.widgets  import RectangleSelector
np.seterr(divide='ignore', invalid='ignore')
from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'inline')
import glob
import time
from silx.image import sift
import silx
import math
from numpy import where
import datetime
from math import sqrt

import glob
import os
import keyboard
from scipy.optimize import curve_fit
import cv2



from PyQt5 import QtWidgets, uic
from pymmcore_plus import CMMCorePlus
from qtpy.QtWidgets import QApplication, QGroupBox, QHBoxLayout, QBoxLayout, QWidget,QGridLayout

from pymmcore_widgets import ExposureWidget
get_ipython().run_line_magic('matplotlib', 'qt5')


############ ****************
#########    ***   !!!    ***

#   Here I am addind the folder with required functions file and then importing *-everything from this file
#   if you want to specify the function that will be imported you need to write the function name intead of *
sys.path.append("Z:\Hanna\CODE")
from functions import *

############ ****************
#########    ***   !!!    ***
pathMM =  "C:\Program Files\Micro-Manager-2.0_NB"   ## path to Micromanager folder on the computer
config = "Phasics_dis_coller_Ni.cfg"                ## name of configuration file. Need to be created in MicroManager 
                                                     # before starting this soft, NikonTi and Phasics camera need to be in devices

#pymmcore.CMMCore().getAPIVersionInfo()              
mmc = CMMCorePlus()                              ##  initialisation of MicroManager core  
mmc.setDeviceAdapterSearchPaths([pathMM])             ##  looking for device adapters  
mmc.loadSystemConfiguration(os.path.join(pathMM, config))   #  Loading configuration

############ ****************
#########    ***   !!!    ***
##   As in our study we used QLSI module i.e. Phasics (Andor) camera, there is the requirement to turn off the sensor cooling 
print('The cooling of camera sensor is :',  mmc.getProperty("Andor sCMOS Camera", "SensorCooling")) ##checking if the cooler is off
mmc.waitForDevice('TIDiaLamp')      ## The lamp of Nikon Ti can take time to turn on so we need to wait a bit

mmc.setAutoShutter(False)    ## 

############ ****************
#########    ***   !!!    ***
lamp = str('TIDiaLamp')
#mmc.getProperty(lamp, "ComputerControl")
mmc.setProperty(lamp, "ComputerControl", "On")   ###  preparing the lamp of Nikon microscope
mmc.setProperty(lamp, "Intensity", 4)
mmc.setProperty(lamp, "State", 1)


Zstage = mmc.getFocusDevice()    # giving the name to the zdrive
exposure = mmc.getExposure()
pos0 = mmc.getPosition(Zstage)   #  getting current position
mmc.setROI(500,500 ,1500, 1500)  #for Phasics camera it is required to make the roi

mmc.snapImage()    # The camera gets the image to have it in the bufffer for further use

#######
##   In this part I define all the functios that will be used in the gui
viewer = napari.Viewer()    ## opening napari viewer 

exposure = 10    ## the deaful value of exposure time

center1_x,center1_y,center2_x,center2_y = 1005, 320, 1170, 1003 # the positions of harmonics of the fourier transform of 1500*1500 image

#textBrow.append("Initialization Done")

global pathth

def start_live():   ##   function to start live\ show images in real time
    textBrow.append("Live is running, to stop/close the window press 'q'")    
    while (True):
        mmc.snapImage()
        frame = mmc.getImage()
        gray = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        gray =  cv2.resize(gray, (700, 700))  ##  to resize the window (otherwise can be bigger than screen)
        cv2.imshow('frame',  gray)
        cv2.setWindowTitle("frame", "Pres q to quit")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

def set_exposure():     ## The function to set new exposure time from input field in the gui
    expos = int(lineExp.text())      ## reading the value from the input field line
    mmc.setExposure(expos)           ## seting exposure time 
    global exposure 
    exposure = mmc.getExposure()     ## reading exposure from MicroManager to be sure thet it was set
    textBrow.append("Exposure time was set to : "+str(exposure))
    return exposure   

def set_lamp_int():   # the function to set lamp intensity to value entered into corresponding line in gui
    global ints   # intensity value set to be global. Allows to access it from other functions
    ints = int(lineLamp.text())   # reading value from the input line
    mmc.setProperty(lamp, "Intensity", ints)   # the intensity set to lamp
    textBrow.append("Nikon lamp intensity was set to "+str(ints))
    return ints
    
def Lamp_on():                      ## the function to turn On the microscope Lamp
    mmc.setProperty(lamp, "State", 1)  
    
def Lamp_off():                     ## the function to turn Off microscope Lamp    
    mmc.setProperty(lamp, "State", 0)

 
def set_z_pos():        #   the function to move z drive  
    pos0 = mmc.getPosition(Zstage)    # getting current position
    pos = np.float32(lineZpos.text())   # reading desired value of movement from gui line
    print("pos = ", pos)
    mmc.setPosition(Zstage, pos+pos0)
    mmc.waitForDevice(Zstage)
    textBrow.append("the position of z-drive is set to = "+str(mmc.getPosition(Zstage)))    


global R 
def reference_image():  
    global reference
    start = time.time()
    textBrow.append("Starting refererence aquisition")
    progressBar.resetFormat()    ## progress bar shows nothing at the begining
    try:   ### To check if path to folder was defined
        pathth
    except NameError:    # in case if path was not defined before it will be read from line in the gui
        pathth = linePath.text() 
    if os.path.isdir(pathth) == False:   ## if path do not exist the error message will pop up
        textBrow.append("Path does not exists, try again")
        return
    else:  
        with open(pathth + '\\METADATA.txt', 'a') as file:     ## opening the Metadata file, if file do not exists it will be created automaticaly
            file.write('Ref recorded at: %s\n Exposure: %s\n Lamp: %s\n' %(datetime.datetime.now(), exposure, ints))
        im =[]
        n_of_im = int(lineRef.text())     ## reading the number of images to acquire from corresponding input field
        c = 100/n_of_im
        mmc.startSequenceAcquisition(n_of_im, 0, True) 
        i=0
        while mmc.isSequenceRunning():
            if mmc.getRemainingImageCount() != 0:
                image_and_MD_raw = mmc.popNextImageAndMD(fix = False)
                image_and_MD = np.asarray([image_and_MD_raw[0], image_and_MD_raw[1].json()], dtype=object)
                im.append(image_and_MD[0])
                progressBar.setValue(int(i*c))
                i=i+1   
        ref = np.mean(np.asarray(im), axis=0)
        viewer.add_image(ref, name='Reference image')
        R = ref.shape[0] // 6
        #start = time.time()
        xm, ym = np.meshgrid(np.arange(ref.shape[1]), np.arange(ref.shape[0]))
        mask_x = np.sqrt((xm - center1_x)**2 + (ym - center1_y)**2) <= R
        mask_y = np.sqrt((xm - center2_x)**2 + (ym - center2_y)**2) <= R
        fourier_ref = fft.fftshift(fft.fft2(ref, norm="ortho")).astype(np.complex64)
        Ix_ref = fft.ifft2(fftpack.fftshift((fourier_ref) * mask_x))
        Iy_ref = fft.ifft2(fftpack.fftshift((fourier_ref) * mask_y))
        reference = np.stack((ref, mask_x, mask_y, Ix_ref, Iy_ref))
        np.save(file_name_check(pathth + '\\ref.npy'), reference.reshape(5*ref.shape[0], ref.shape[1]))
        textBrow.append('The obtained images were saved under the name ref.tif')
        
        #return image_ref
    textBrow.append('time:' +str(time.time()-start))
    with open(pathth + '\\METADATA.txt', 'a') as file:
        file.write('The number of images avaraged for reference: %s\n' %n_of_im)
    textBrow.append("Done")
    return 


def stack_acquisition():  # the function to perform the acqusition without autofocusing
    progressBar.resetFormat()   
    try:
        pathth
    except NameError:
        pathth = linePath.text() 
    if os.path.isdir(pathth) == False:
        textBrow.append("Path does not exists, try again")
        return
    else:    
        im =[]
        stack = []
        nIm = int(line_acq.text())
        if line_nIm.text() =='':
            nImAv = 10
        else:
            nImAv = int(line_nIm.text())
        c = 100/(nIm*nImAv)
        mmc.startSequenceAcquisition(nIm*nImAv, 0, True) 
        i=0
        while mmc.isSequenceRunning():
            if mmc.getRemainingImageCount() != 0:
                image_and_MD_raw = mmc.popNextImageAndMD(fix = False)
                image_and_MD = np.asarray([image_and_MD_raw[0], image_and_MD_raw[1].json()], dtype=object)
                im.append(image_and_MD[0])
                i=i+1
                progressBar.setValue(int(i*c))
            if keyboard.is_pressed('q'):  # Check if 'q' is pressed
                textBrow.append("Interrupted by user")
                break  
            if keyboard.is_pressed('s'):  # Check if 'q' is pressed
                textBrow.append("Interrupted by user, the data will be saved")
                images = np.asarray(im)
                Stack = np.asarray([np.mean(images[i:i+nImAv], axis=0) for i in range(0, len(images), nImAv)]) 
                textBrow.append("Saving the data...")
                save_path = file_name_check(pathth + '\\Stack.tif')
                _, tail = os.path.split(save_path)
                imwrite(save_path, Stack.astype(np.float16))
                textBrow.append("Done")
                break 
        images = np.asarray(im)
        Stack = np.asarray([np.mean(images[i:i+nImAv], axis=0) for i in range(0, len(images), nImAv)]) 
        progressBar.setFormat("Saving")
        save_path = file_name_check(pathth + '\\Stack.tif')
        _, tail = os.path.split(save_path)
        imwrite(save_path, Stack.astype(np.float16))
        textBrow.append('The obtained images were saved under the name ' +srt(tail))
        textBrow.append("Done")
        
#global zInt
def zStack():
    global reference
    progressBar.resetFormat()
    start = time.time()
    try:   ### To check if path to folder was defined
        pathth
    except NameError:    # in case if path was not defined before the path will be read from line in the gui
        pathth = linePath.text() 
    if os.path.isdir(pathth) == False:   ## if path do not exist the error message will pop up
        textBrow.append("Path does not exists, try again")
        return
    else:
        if 'reference' not in globals():
            print('reading zGradients from disk')
            list_of_files = glob.glob(os.path.normpath(pathth+ '\\*'))
            refpath = os.path.normpath(max([file for file in list_of_files if 'ref' in file], key=os.path.getmtime))
            reference = np.load(refpath)
            reference = reference.reshape(5, int(reference.shape[0]/5), reference.shape[1])
        ref = reference[0].real
        Zstage = mmc.getFocusDevice()
        if line_start.text() =='':
            start_pos = -1.5
        else:
            start_pos = np.float32(line_start.text())
        if line_stop.text() =='':
            stop_pos = 1.5
        else:
            stop_pos = np.float32(line_stop.text())
        if line_step.text() =='':
            step = 0.05
        else:
            step = np.float32(line_step.text())
        pos0 = mmc.getPosition(Zstage)
        print("0 position : ", pos0)
        if line_nIm.text() =='':
            nIm = 10
        else:
            nIm = int(line_nIm.text())
        start_pos = start_pos+pos0
        stop_pos = stop_pos + pos0 #+ step
        print("start:", start_pos, "stop: ", stop_pos, "step :", step)
        mmc.snapImage()
        length = len(np.arange(start_pos, stop_pos, step))
        print(length)
        c = 100/length 
        frame = 0
        posR = []
        stack=[]
        with open(pathth + '\\METADATA.txt', 'a') as file:
            file.write('zStack recorded at %s\n zStack \n start:%s\n end: %s\n step %s\n N of averaged images %s\n Exposure: %s\n Lamp: %s\n'  %(datetime.datetime.now(), start_pos, stop_pos, step, nIm, exposure, ints))
        for pos in np.arange(start_pos, stop_pos, step):
            textBrow.append("pos = "+str(pos))
            mmc.setPosition(Zstage, pos)
            mmc.waitForDevice(Zstage)
            mmc.startSequenceAcquisition(nIm, 0, True)
            if keyboard.is_pressed('q'):  # Check if 'q' is pressed
                textBrow.append("Interrupted by user")
                break  
            while mmc.isSequenceRunning():
                if mmc.getRemainingImageCount() != 0:
                    image_and_MD_raw = mmc.popNextImageAndMD(fix = False)
                    image_and_MD = np.asarray([image_and_MD_raw[0], image_and_MD_raw[1].json()], dtype=object)
                    stack.append(image_and_MD[0])
            textBrow.append("pos read = "+str(mmc.getPosition(Zstage)))
            posR.append(mmc.getPosition(Zstage))
            frame = frame + 1 
            progressBar.setValue(int(frame*c))
        textBrow.append('Saving, it took:'+str(time.time()-start))
        z_Stack = np.asarray([np.mean(stack[i:i+nIm], axis=0) for i in range(0, len(stack), nIm)])
        #viewer.add_image(z_Stack, name = 'z Stack')
        global zGrads, zInt
        zGrads = grad_diff(z_Stack, reference[1], reference[2],reference[3],reference[4])
        viewer.add_image(zGrads, name = 'gradients z Stack')
        mmc.setPosition(Zstage, pos0)
        np.save(file_name_check(pathth + '\\zStackRaw.npy'), z_Stack.astype(np.float16).reshape(z_Stack.shape[0]*z_Stack.shape[1], z_Stack.shape[2]))
        zInt = intensity_image(z_Stack, ref, choose_coord=False)
        np.savetxt(file_name_check(pathth + '\\z_posRead.txt'), posR)
        textBrow.append('The obtained images were saved')
        progressBar.setFormat("Done")
    mmc.setPosition(Zstage, pos0)
    #viewer.add_image(phase_image(z_Stack[int(len(z_Stack)/2)], ref, choose_coord=False), name = 'Phase z Stack')
    print('Done', 'pos now = ', pos0, 'it took: ', time.time()-start )
    return zGrads

def start_live_grad(): ##   function to start live\ show images in real time
    try:
        reference
    except NameError:
        try:
            ref = imread(pathth+'\\ref.tif' )
        except: 
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Information)
            msgBox.setText("There is no reference")
            msgBox.setStandardButtons(QMessageBox.Ok)
            msgBox.exec_()  
    while (True):
        mmc.snapImage()
        frame = mmc.getImage()
        diff = grad_diff(frame, mask_x, mask_y,Ix_ref,Iy_ref )
        gray = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
        gray =  cv2.resize(gray, (1000, 1000))     ##  to resize the window (otherwise can be bigger than screen)
        cv2.imshow('frame',  gray)
        cv2.setWindowTitle("frame", "Pres q to quit")
        mmc.sleep(exposure)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

def check_max():
    mmc.snapImage()
    image = mmc.getImage()
    textBrow.append("The maaximum value on the image is "+str(image.max()))
    

def test_z_stack():
    global reference
    progressBar.resetFormat()
    start = time.time()
    textBrow.append("Starting zStack acquisition")
    try:   ### To check if path to folder was defined
        pathth
    except NameError:    # in case if path was not defined before the path will be read from line in the gui
        pathth = linePath.text() 
    if os.path.isdir(pathth) == False:   ## if path do not exist the error message will pop up
        textBrow.append("Path does not exists, try again")
        return
    else:
        if 'reference' not in globals():
            textBrow.append('reading ref from disk')
            list_of_files = glob.glob(os.path.normpath(pathth+ '\\*'))
            refpath = os.path.normpath(max([file for file in list_of_files if 'ref' in file], key=os.path.getmtime))
            reference = np.load(refpath)
            reference = reference.reshape(5, int(reference.shape[0]/5), reference.shape[1])
        ref = reference[0].real
        Zstage = mmc.getFocusDevice()
        if line_start.text() =='':  # if the value is not specified in GUI, the default one will be used
            start_pos = -1.5
        else:
            start_pos = np.float32(line_start.text())
        if line_stop.text() =='':
            stop_pos = 1.5
        else:
            stop_pos = np.float32(line_stop.text())
        if line_step.text() =='':
            step = 0.05
        else:
            step = np.float32(line_step.text())
        pos0 = mmc.getPosition(Zstage)
        textBrow.append("0 position : "+str(pos0))
        if line_nIm.text() =='':
            nIm = 8
        else:
            nIm = int(line_nIm.text())
        start_pos = start_pos+pos0
        stop_pos = stop_pos + pos0 
        textBrow.append("start:"+str(start_pos)+"stop: "+str(stop_pos)+ "step :"+str(step))
        mmc.snapImage()
        length = len(np.arange(start_pos, stop_pos, step))
        textBrow.append(str(length)+"of zStack")
        c = 100/length 
        frame = 0
        posR = []
        stack=[]
        with open(pathth + '\\METADATA.txt', 'a') as file:
            file.write('zStack recorded at %s\n zStack \n start:%s\n end: %s\n step %s\n N of averaged images %s\n Exposure: %s\n Lamp: %s\n'  %(datetime.datetime.now(), start_pos, stop_pos, step, nIm, exposure, ints))
        pos_list = []
        pos=start_pos
        rpos = pos-pos0
        while rpos < stop_pos-pos0:
            pos_list.append(pos)
            if (rpos>-0.4) and (rpos <0.4):
                print('step = 0.05:' , pos)
                pos=pos+step
            elif (rpos<-0.4)or(rpos>0.4):
                print('step = 0.2', pos)
                pos=pos+0.2
            elif (rpos<-0.4)or(rpos>0.4):
                print('step = 0.2', pos)
                pos=pos+0.2
            rpos=pos-pos0
            if keyboard.is_pressed('q'):  # Check if 'q' is pressed
                textBrow.append("Interrupted by user")
                break  
            textBrow.append("pos = "+str(pos))
            mmc.setPosition(Zstage, pos)
            mmc.waitForDevice(Zstage)
            mmc.startSequenceAcquisition(nIm, 0, True)
            while mmc.isSequenceRunning():
                if mmc.getRemainingImageCount() != 0:
                    image_and_MD_raw = mmc.popNextImageAndMD(fix = False)
                    image_and_MD = np.asarray([image_and_MD_raw[0], image_and_MD_raw[1].json()], dtype=object)
                    stack.append(image_and_MD[0])
            textBrow.append("pos read = "+str(mmc.getPosition(Zstage)))
            posR.append(mmc.getPosition(Zstage))
            frame = frame + 1 
            progressBar.setValue(int(frame*c))
            if keyboard.is_pressed('q'):  # Check if 'q' is pressed
                textBrow.append("Interrupted by user")
                break  
        z_Stack = np.asarray([np.mean(stack[i:i+nIm], axis=0) for i in range(0, len(stack), nIm)])
        mmc.setPosition(Zstage, pos0)
        #####################################     To Rytov/Intensity part
        textBrow.append('acquired'+str(time.time()-start))
        #zInt = intensity_image(z_Stack, ref, choose_coord=False)
        global zGrads, zInt
        zGrads = grad_diff(z_Stack, reference[1], reference[2],reference[3],reference[4])
        viewer.add_image(zGrads, name = 'grqdients z Stack')
        textBrow.append('stacks done'+str( time.time()-start))
        textBrow.append("Saving the data ...")
        mmc.setPosition(Zstage, pos0)
        np.save(file_name_check(pathth + '\\zStackRaw.npy'), z_Stack.astype(np.float16).reshape(z_Stack.shape[0]*z_Stack.shape[1], z_Stack.shape[2]))
        zInt = intensity_image(z_Stack, ref, choose_coord=False)
        np.savetxt(file_name_check(pathth + '\\z_posRead.txt'), posR)
        textBrow.append('Done,  pos now = '+str(pos0)+'time = '+str(time.time()-start))
        msgBox = QMessageBox()
        msgBox.setText('The obtained images were saved')
        msgBox.exec_()
        progressBar.setFormat("Done")
    mmc.setPosition(Zstage, pos0)
    viewer.add_image(phase_image(z_Stack[int(len(z_Stack)/2)], ref, choose_coord=False), name = 'Example of phase image from z Stack')
    


def acquisition_calib_grad():  ## the function to perform autofocusing with gradient images
    progressBar.resetFormat()   ## 
    textBrow.append("Starting acquisition with autofocusing using Gradient images")  #message to GUI window
    try:   ### To check if path to folder was defined
        pathth
    except NameError:    # in case if path was not defined before the path will be read from line in the gui
        pathth = linePath.text() 
    if os.path.isdir(pathth) == False:   ## if path do not exist the error message is shown
        textBrow.append("Path does not exists, try again")   # message to GUI wimdow
    list_of_files = glob.glob(os.path.normpath(pathth+ '\\*'))   # list of files in folder
    refpath = os.path.normpath(max([file for file in list_of_files if 'ref' in file], key=os.path.getmtime)) # looking for the last file with name containing 'ref'
    reference = np.load(refpath)  # saving path to  last ref file 
    Reference = reference.reshape(5, int(reference.shape[0]/5), reference.shape[1]) # geting ref image frm ref file
    ref = Reference[0].real  # .real because ref can be saved in complex format 
    mask_x = Reference[1] 
    mask_y = Reference[2]
    Ix_ref = Reference[3]
    Iy_ref = Reference[4]
    Gradpath = os.path.normpath(max([file for file in list_of_files if 'zStack' in file], key=os.path.getmtime)) # geting path to last gradients zStack
    _, tail = os.path.split(Gradpath) 
    if 'zGrads' not in globals():  # checking if Gradients z Stack was saved in global values, if no  read from disk
        textBrow.append('reading zGradients from disk') 
        zImage = np.load(Gradpath)
        zImage = zImage.reshape(int(zImage.shape[0]/ref.shape[0]), ref.shape[0], ref.shape[1])
        zImage = grad_diff(zImage, Reference[1], Reference[2],Reference[3],Reference[4])
    else:
        zImage = zGrads
    textBrow.append('The start position'+str(mmc.getPosition(Zstage)))
    nIm = int(lineAcqCalibNbI.text())  # getting the numebr of images to acquire from GUI
    nImAv = 8 #int(line_nImAcq.text())
    if lineFitDegree.text() =='':  # if degree was not specified it will be set to 12
        degree = 12# = 10
    else:
        degree = int(lineFitDegree.text()) # the values will be used  if specified in GUI
    frequency = int(lineAcqCalibFr.text()) # getting frequency of refocusing from GUI
    Stack = np.zeros((nIm, int(mmc.getImage().shape[0]), int(mmc.getImage().shape[1]))) # empty array where images will be added
    
    posPath = os.path.normpath(max([file for file in list_of_files if '_posRead' in file], key=os.path.getmtime))
    poslist = np.loadtxt(posPath)
    cfr = 1
    drft = []
    posN=[]
    micPos=[]
    poslist = poslist-poslist.min()
    x = poslist-poslist[int(len(poslist)/2)]
    xx = np.arange(x.min(), x.max(), 0.001)
    frame = int(len(zImage)/2)
    sift_ocl = silx.image.sift.SiftPlan(template=zImage[frame], devicetype="GPU") # zImage[frame] will be used as template for xy correction
    keypoints = sift_ocl(zImage[frame]) #looking for keypoints on the image
    mp = sift.MatchPlan() 
    # creating the figure where the calculated drift values will be displayed
    plt.ion()
    xg = np.linspace(0, nIm, nIm)
    yg = np.arange(-1.5, 1.5, 3/nIm)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    line1, = ax.plot(xg, yg, 'b+:')
    plt.xlabel("Refocusing step")
    plt.ylabel("Drift, μm")
    plt.title("Updating plot...") 
    yyg = np.zeros(nIm)
    yyg[:]= np.nan
    yygg = np.zeros(nIm)
    yygg[:]= np.nan
    k=0
    # creating the figure where the values read from z-drive will be displayed
    plt.ion()
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    line2, = ax2.plot(x, np.linspace(0, 0.0005, len(zImage)), 'b+:')
    line3, = ax2.plot(x, np.linspace(0, 0.0005, len(zImage)), 'r')
    plt.xlabel("Refocusing step")
    plt.ylabel("CCC, μm")
    plt.title("Updating plot...") 
    progressBar.setValue(0)
    c = 100/nIm
    for i in range(0, nIm): # starting the main loop of acquisition with autofocusing
        if keyboard.is_pressed('q'):  # Check if 'q' is pressed
            textBrow.append("Interrupted by user")
            break   
        if keyboard.is_pressed('s'):  # Check if 's' is pressed if yes the process is interupted while saving the data
            textBrow.append("Interrupted by user, the data will be saved")
            np.savetxt(file_name_check(pathth+'/detectedDrift_grad_interupted_z'+tail[-5]+'.txt'), pd.concat([pd.DataFrame(drft), pd.DataFrame(micPos)], axis=1))
            progressBar.setFormat("Done, interupted")
            np.save(file_name_check(pathth + '\\Stack_autofocus_grad_interupted_z'+tail[-5]+'.npy'), Stack.astype(np.float16).reshape(Stack.shape[0]*Stack.shape[1], Stack.shape[2]))
            textBrow.append("Done")
            break  
        mmc.snapImage() # getting the image from camera
        stack = np.zeros((nImAv, int(mmc.getImage().shape[0]), int(mmc.getImage().shape[1]))) # images will be stored to stack 
        for n in range(0, nImAv): # getting 10 images to avarage
            mmc.snapImage()
            stack[n] = mmc.getImage()
        st = np.mean(stack, axis=0) # getting mean value of images
        Stack[i] = st  # mean image stored in Stack
        ###############################
        if (i % frequency == 0) == True: # when the frequency matches the value specified in GUI do autofocusing
            if lineFitDegree.text() =='':  # checking if degree of fit was changed in GUI
                degree = 12# = 10
            else:
                degree = int(lineFitDegree.text())
            corrcoef = [] 
            pos = mmc.getPosition(Zstage)  # current position of z-drive
            start = time.time()  # getting current time
            imm = grad_diff(Stack[i], mask_x, mask_y,Ix_ref,Iy_ref) # calculation of gradient image
            ### Shift correction
            if keyboard.is_pressed('q'):  # Check if 'q' is pressed
                textBrow.append("Interrupted by user")
                break   
            try: 
                im_keypoints = sift_ocl(imm.astype(np.float32)) # looking for keyposint on the new image
                match = mp(keypoints, im_keypoints) # looking for maching keyposint
                sa = silx.image.sift.LinearAlign(zImage[frame], devicetype="GPU") # xy-drift numerical correction
                im = sa.align(imm,  shift_only=True)
            except: # if something went wrong the process is interupted and data is saved
                np.savetxt(file_name_check(pathth+'/detectedDrift_grad_err1_z'+tail[-5]+'.txt'), pd.concat([pd.DataFrame(drft), pd.DataFrame(micPos)], axis=1))
                textBrow.append("Done, data saved after xy-drfit correction didn't work")
                imwrite(file_name_check(pathth + '\\Stack_autofocus_grad_err1_z'+tail[-5]+'.tif', Stack.astype(np.float16)))
                with open(pathth + '\\METADATA.txt', 'a') as file:
                    file.write('Aqqusition with autofocus recorded FAIL xy correction at %s\n N of images %s \n frequency:%s\n Exposure: %s\n Lamp: %s\n degree: %s\n ' %(datetime.datetime.now(), nIm, frequency, exposure, ints, degree))
                textBrow.append('xy correction failed, data was saved')
                break
            ######################
            for f in range(0, len(zImage)):  # loop to run through all images in zStack and calculate cross-correlation coeficients (CCC)
                try:    
                    corrcoef.append(ncc(im[200:1300, 200:1300], zImage[f,200:1300, 200:1300])) ## 
                except: # if something went wrong interupt with saving data
                    np.savetxt(file_name_check(pathth+'/detectedDrift_grad_err2_z'+tail[-5]+'.txt'), pd.concat([pd.DataFrame(drft), pd.DataFrame(micPos)], axis=1))
                    textBrow.append("Done, data saved after xy-drfit correction didn't work")
                    imwrite(file_name_check(pathth + '\\Stack_autofocus_grad_err2_z'+tail[-5]+'.tif'), Stack.astype(np.float16))
                    with open(pathth + '\\METADATA.txt', 'a') as file:
                        file.write('Aqqusition with autofocus recorded FAIL correlation at %s\n N of images %s \n frequency:%s\n Exposure: %s\n Lamp: %s\n degree: %s\n ' %(datetime.datetime.now(), nIm, frequency, exposure, ints, degree))
                    textBrow.append('correlation failed, data was saved')
                    break
            y = np.array(corrcoef) 
            model2 = np.poly1d(np.polyfit(x, y, degree)) # fitting the cross-correlation curve
            drft.append(xx[where(model2(xx) == model2(xx).max())])  # the calculated drift stored into 'drft' list
            i=i+1
            yyg[i] = drft[-1][0]  # updayting the value on the figure
            line1.set_ydata(yyg)
            k = k+1
            fig.canvas.draw()   
            fig.canvas.flush_events()
            ax2.set_ylim(y.min()-y.min()*0.1, y.max()+y.max()*0.1)
            line2.set_ydata(y)
            line3.set_ydata(model2(x))
            fig2.canvas.draw()   
            fig2.canvas.flush_events()
            textBrow.append('newvalue'+str(drft[-1]))  # message to GIU with the last value of drift calculated
            if keyboard.is_pressed('q'):  # Check if 'q' is pressed
                textBrow.append("Interrupted by user")
                break     
            micPos.append(mmc.getPosition(Zstage)) # reading current position of z-drive and storing into list
            if keyboard.is_pressed('s'):  # Check if 'q' is pressed
                textBrow.append("Interrupted by user, the data will be saved")
                np.savetxt(file_name_check(pathth+'/detectedDrift_grad_interupted_z'+tail[-5]+'.txt'), pd.concat([pd.DataFrame(drft), pd.DataFrame(micPos)], axis=1))
                progressBar.setFormat("Done, interupted")
                np.save(file_name_check(pathth + '\\Stack_autofocus_grad_interupted_z'+tail[-5]+'.npy'), Stack.astype(np.float16).reshape(Stack.shape[0]*Stack.shape[1], Stack.shape[2]))
                break 
            if (np.abs(drft[-1]) < 0.010): # if the drift was smaller then 10 nm corresction is not performed to avoid werd behavious of z-drive
                textBrow.append('drift was smaller then 10 nm no correction is done')
                continue 
            mmc.setPosition(Zstage, pos-drft[-1][0]) # the new position (corrected one) is set to z-drive 
            mmc.waitForDevice(Zstage) # to ensure that z-drive finisehd movement
            cfr = cfr+1
            textBrow.append("calibration, detected drift = "+str(drft[-1])+'the drift was = '+str(pos-drft[-1])+'microscope reply = '+str(mmc.getPosition(Zstage))+'it took '+str(time.time()-start)+'to refocus')   
            progressBar.setValue(int(i*c))
    plt.figure()  # after acquisition is finished the last cross-correlation curve and fit are displayed 
    plt.plot(xx, model2(xx))
    plt.plot(x, y)
    plt.figure()
    plt.plot(micPos)
    plt.show()
    np.savetxt(file_name_check(pathth+'/detectedDrift_grad_z'+tail[-5]+'.txt'), pd.concat([pd.DataFrame(drft), pd.DataFrame(micPos)], axis=1))
    textBrow.append("Done")
    np.save(file_name_check(pathth + '\\Stack_autofocus_grad_z'+tail[-5]+'.npy'), Stack.astype(np.float16).reshape(Stack.shape[0]*Stack.shape[1], Stack.shape[2]))
    with open(pathth + '\\METADATA.txt', 'a') as file:
        file.write('Aqqusition with autofocus recorded at %s\n N of images %s \n frequency:%s\n Exposure: %s\n Lamp: %s\n degree: %s\n' %(datetime.datetime.now(), nIm, frequency, exposure, ints, degree))

  

def acquisition_calib_int():
    progressBar.resetFormat()
    textBrow.append("Starting acquisition with active autofocusing using Intensity images")
    try:   ### To check if path to folder was defined
        pathth
    except NameError:    # in case if path was not defined before the path will be read from line in the gui
        pathth = linePath.text() 
    if os.path.isdir(pathth) == False:   ## if path do not exist the error message will pop up
        textBrow.append("Path does not exists, try again")
    list_of_files = glob.glob(os.path.normpath(pathth+ '\\*'))
    if 'reference' not in globals(): 
        textBrow.append('reading reference from drive') 
        refpath = os.path.normpath(max([file for file in list_of_files if 'ref' in file], key=os.path.getmtime))
        reference = np.load(refpath)
        reference = reference.reshape(5, int(reference.shape[0]/5), reference.shape[1])
    else:
        reference
        Reference = reference
    ref = Reference[0].real
    mask_x = Reference[1]
    mask_y = Reference[2]
    Ix_ref = Reference[3]
    Iy_ref = Reference[4]
    Gradpath = os.path.normpath(max([file for file in list_of_files if 'zStack' in file], key=os.path.getmtime))
    _, tail = os.path.split(Gradpath)
    if 'zInt' not in globals():
        textBrow.append('reading zGradients from disk')
        zImage = np.load(Gradpath)
        zImage = zImage.reshape(int(zImage.shape[0]/ref.shape[0]), ref.shape[0], ref.shape[1])
        zImage = intensity_image(zImage, ref, choose_coord=False)
    else:
        zImage = zInt
    textBrow.append('The start position'+str(mmc.getPosition(Zstage)))
    nIm = int(lineAcqCalibNbI.text())
    nImAv = 10 
    if lineFitDegree.text() =='':
        degree = 12# = 10
    else:
        degree = int(lineFitDegree.text())
    frequency = int(lineAcqCalibFr.text())
    Stack = np.zeros((nIm, int(mmc.getImage().shape[0]), int(mmc.getImage().shape[1])))
    
    posPath = os.path.normpath(max([file for file in list_of_files if '_posRead' in file], key=os.path.getmtime))
    poslist = np.loadtxt(posPath)
    cfr = 1
    drft = []
    posN=[]
    micPos=[]
    poslist = poslist-poslist.min()
    x = poslist-poslist[int(len(poslist)/2)]
    xx = np.arange(x.min(), x.max(), 0.001)
    frame = int(len(zImage)/2)
    sift_ocl = silx.image.sift.SiftPlan(template=zImage[frame], devicetype="GPU")
    keypoints = sift_ocl(zImage[frame])
    mp = sift.MatchPlan()
    
    plt.ion()
    xg = np.linspace(0, nIm, nIm)
    yg = np.arange(-1.5, 1.5, 3/nIm)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    line1, = ax.plot(xg, yg, 'b+:')
    plt.xlabel("Refocusing step")
    plt.ylabel("Drift, μm")
    plt.title("Updating plot...") 
    yyg = np.zeros(nIm)
    yyg[:]= np.nan
    yygg = np.zeros(nIm)
    yygg[:]= np.nan
    k=0
    
    plt.ion()
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    line2, = ax2.plot(x, np.linspace(0, 0.0005, len(zImage)), 'b+:')
    line3, = ax2.plot(x, np.linspace(0, 0.0005, len(zImage)), 'r')
    plt.xlabel("Refocusing step")
    plt.ylabel("CCC, μm")
    plt.title("Updating plot...") 
    progressBar.setValue(0)
    c = 100/nIm
    for i in range(0, nIm):
        if keyboard.is_pressed('q'):  # Check if 'q' is pressed
            textBrow.append("Interrupted by user")
            break  
        mmc.snapImage()
        stack = np.zeros((nImAv, int(mmc.getImage().shape[0]), int(mmc.getImage().shape[1])))
        for n in range(0, nImAv):
            mmc.snapImage()
            stack[n] = mmc.getImage()
        st = np.mean(stack, axis=0)
        Stack[i] = st  
        ###############################
        if (i % frequency == 0) == True:
            ###############################
            if lineFitDegree.text() =='':
                degree = 12
            else:
                degree = int(lineFitDegree.text())
            corrcoef = []
            pos = mmc.getPosition(Zstage)
            start = time.time()
            imageInt = intensity_image(Stack[i], ref, choose_coord = False)
            ### Shift correction
            try: 
                im_keypoints = sift_ocl(imageInt.astype(np.float32))
                match = mp(keypoints, im_keypoints)
                sa = silx.image.sift.LinearAlign(zImage[frame], devicetype="GPU")
                im = sa.align(imageInt,  shift_only=True)
            except:
                np.savetxt(file_name_check(pathth+'/detectedDrift_int_err1_z'+tail[-5]+'.txt'), pd.concat([pd.DataFrame(drft), pd.DataFrame(micPos)], axis=1))
                imwrite(file_name_check(pathth + '\\Stack_autofocus_int_err1_z'+tail[-5]+'.tif', Stack.astype(np.float16)))
                with open(pathth + '\\METADATA.txt', 'a') as file:
                    file.write('Finished with xy drift correction error, aqusition with autofocus recorded at %s\n N of images %s \n frequency:%s\n Exposure: %s\n Lamp: %s\n degree: %s\n ' %(datetime.datetime.now(), nIm, frequency, exposure, ints, degree))
                textBrow.append('xy correction failed, data was saved')
                break
            ######################
            for f in range(0, len(zImage)):
                try:    
                    corrcoef.append(ncc(im[200:1300, 200:1300], zImage[f,200:1300, 200:1300]))
                except:
                    np.savetxt(file_name_check(pathth+'/detectedDrift_grad_err2_z'+tail[-5]+'.txt'), pd.concat([pd.DataFrame(drft), pd.DataFrame(micPos)], axis=1))
                    textBrow.append("Done, data saved after xy-drfit correction didn't work")
                    imwrite(file_name_check(pathth + '\\Stack_autofocus_grad_err2_z'+tail[-5]+'.tif'), Stack.astype(np.float16))
                    with open(pathth + '\\METADATA.txt', 'a') as file:
                        file.write('Aqqusition with autofocus recorded FAIL correlation at %s\n N of images %s \n frequency:%s\n Exposure: %s\n Lamp: %s\n degree: %s\n ' %(datetime.datetime.now(), nIm, frequency, exposure, ints, degree))
                    textBrow.append('xy correction failed, data was saved')
                    break
            y = np.array(corrcoef)
            model2 = np.poly1d(np.polyfit(x, y, degree))
            drft.append(xx[where(model2(xx) == model2(xx).max())])
            i=i+1
            yyg[i] = drft[-1][0]
            line1.set_ydata(yyg)
            k = k+1
            fig.canvas.draw()   
            fig.canvas.flush_events()
            ax2.set_ylim(y.min()-y.min()*0.1, y.max()+y.max()*0.1)
            line2.set_ydata(y)
            line3.set_ydata(model2(x))
            fig2.canvas.draw()   
            fig2.canvas.flush_events()
            if keyboard.is_pressed('q'):  # Check if 'q' is pressed
                textBrow.append("Interrupted by user")
                break  
            textBrow.append('newvalue'+str(drft[-1]))
            micPos.append(mmc.getPosition(Zstage))
            if keyboard.is_pressed('s'):  # Check if 'q' is pressed
                textBrow.append("Interrupted by user, the data will be saved")
                np.savetxt(file_name_check(pathth+'/detectedDrift_int_interupted_z'+tail[-5]+'.txt'), pd.concat([pd.DataFrame(drft), pd.DataFrame(micPos)], axis=1))
                progressBar.setFormat("Done, interupted")
                np.save(file_name_check(pathth + '\\Stack_autofocus_int_interupted_z'+tail[-5]+'.npy'), Stack.astype(np.float16).reshape(Stack.shape[0]*Stack.shape[1], Stack.shape[2]))
                break 
            if (np.abs(drft[-1]) < 0.010):
                print('drift was smaller then 10 nm')
                continue 
            mmc.setPosition(Zstage, pos-drft[-1][0])
            mmc.waitForDevice(Zstage)
            cfr = cfr+1
            textBrow.append("calibration, detected drift = "+str(drft[-1])+'the drift was = '+str(pos-drft[-1])+'microscope reply = '+str(mmc.getPosition(Zstage))+'it took '+str(time.time()-start)+'to refocus')   
            progressBar.setValue(int(i*c))
    plt.figure()
    #plt.plot(xx, model2(xx))
    plt.plot(x, y)
    plt.figure()
    plt.plot(micPos)
    plt.show()
    np.savetxt(file_name_check(pathth+'/detectedDrift_int_z'+tail[-5]+'.txt'), pd.concat([pd.DataFrame(drft), pd.DataFrame(micPos)], axis=1))
    textBrow.append("Done")
    np.save(file_name_check(pathth + '\\Stack_autofocus_int_z'+tail[-5]+'.npy'), Stack.astype(np.float16).reshape(Stack.shape[0]*Stack.shape[1], Stack.shape[2]))
    with open(pathth + '\\METADATA.txt', 'a') as file:
        file.write('Aqqusition with autofocus recorded at %s\n N of images %s \n frequency:%s\n Exposure: %s\n Lamp: %s\n degree: %s\n ' %(datetime.datetime.now(), nIm, frequency, exposure, ints, degree))


################################################################################################
################################################################################################
########################    Main window created by QtDesigner   ################################
################################################################################################


class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()
        ui = uic.loadUi('Z:/Hanna/CODE/Autofocus_fArtcl/Autofocus.ui', self)
        global lineRef,lineLamp,line_name,lineExp,line_start,line_stop,line_step,linePath,line_nImAcq, lineZpos,lineDegree   ##  The global names are defined in order
        global line_nIm,progressBar,line_acq,linePhase,lineRoi,lineCalib,labelRefIm,lineRefIm,lineAcqCalibNbI,lineAcqCalibFr,lineFitDegree          # be able to use it outside the window function 
        global checkBoxIntsave,textBrow
        lineFitDegree = ui.fitDegree
        lineRef = ui.lineRef
        lineLamp = ui.lineLamp
        line_nImAcq = ui.line_nImAcq
        #line_name = ui.line_name
        lineExp = ui.lineExp
        line_start = ui.line_start
        line_stop = ui.line_stop
        line_step = ui.line_step
        linePath = ui.line_path
        line_nIm = ui.line_nIm
        progressBar = ui.progressBar
        lineAcqCalibNbI = ui.lineAcqCalibNbI
        lineAcqCalibFr = ui.lineAcqCalibFr
        line_acq = ui.line_acq
        lineZpos = ui.lineZpos
        #checkBoxIntsave = ui.checkBoxIntsave
        #checkBoxRAWsave = ui.checkBoxRAWsave

        textBrow = ui.fuckingText

        #checkBoxIntsave.stateChanged.connect(checkbox_saveInt)
        #checkBoxRAWsave.stateChanged.connect(checkbox_saveRAW)
        
        buttonMaxV = ui.buttonMaxV
        buttonMaxV.clicked.connect(check_max)   
        
        buttonRef = ui.buttonRef
        buttonRef.clicked.connect(reference_image) 
        
        buttonExp = ui.buttonExp
        buttonExp.clicked.connect(set_exposure)
        
        buttonLampOn = ui.LampOn
        buttonLampOn.clicked.connect(Lamp_on)

        buttonLampOff = ui.LampOff
        buttonLampOff.clicked.connect(Lamp_off)
        
        buttonLamp = ui.buttonLamp
        buttonLamp.clicked.connect(set_lamp_int)
        
        buttonZpos = ui.buttonZpos
        buttonZpos.clicked.connect(set_z_pos)
        
        buttonZ = ui.buttonZ
        buttonZ.clicked.connect(zStack)
        
        buttonAcq = ui.buttonAcq
        buttonAcq.clicked.connect(stack_acquisition)
        
        buttonAcqCalib_int = ui.buttonAcqCalib_int
        buttonAcqCalib_int.clicked.connect(acquisition_calib_int)
        
        buttonAcqCalib_grads = ui.buttonAcqCalib_grads
        buttonAcqCalib_grads.clicked.connect(acquisition_calib_grad)
        ########    tests
        buttonZ_stepping = ui.buttonZ_stepping
        buttonZ_stepping.clicked.connect(test_z_stack)
        
        buttonLive = ui.startLive
        buttonLive.clicked.connect(start_live)

        self.show()      
app = QtWidgets.QApplication(sys.argv)
window = Ui()
window.show()
app.exec_()


################################################################################################
################################################################################################


