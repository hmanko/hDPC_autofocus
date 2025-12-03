#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 16:33:36 2025

@author: hannamanko
"""

from PyQt5 import QtWidgets, uic#,QtCore
#import cv2
import numpy as np
import pyqtgraph as pg
from pyqtgraph import ImageView
from PyQt5.QtWidgets import QLabel,QMainWindow,QVBoxLayout, QFileDialog, QWidget
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot #,QThreadPool
import sys
from tifffile import imread, imwrite, TiffWriter
import matplotlib.pyplot as plt

import pymmcore
from pymmcore_plus import CMMCorePlus
import os
import time
from silx.image import sift
import pandas as pd
from numpy import where
import glob

import threading
mmc_lock = threading.Lock()

#from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'inline')
#get_ipython().run_line_magic('matplotlib', 'qt5')
import numpy as np, queue, time
'''


q = queue.Queue()
writer = TiffWriterThread(q, "D:/Data_Programs/tests/acq__0000.tif")
writer.start()

# Simulate 10 frames
for i in range(10):
    q.put(mmc.getImage(mmc.snapImage()))
    time.sleep(0.05)

writer.stop()
writer.join()
'''

pathMM =  "C:/Program Files/Micro-Manager-2.0"   ## path to Micromanager folder on the computer
config = "Hamamatsu_Arduino_Piezo.cfg" 
#config = "MMConfig_demo.cfg" #  "ArduinoKuro.cfg" #♠"PiezoArduino_Kuro.cfg" #                 ## name of configuration file
#pymmcore.CMMCore().getAPIVersionInfo()
mmc = CMMCorePlus.instance()                             ##  initialisation of MicroManager core
mmc.setDeviceAdapterSearchPaths([pathMM])             ##  looking for device adapters
mmc.loadSystemConfiguration(os.path.join(pathMM, config))   #  Loading configuration
mmc.mda.engine.use_hardware_sequencing = True

mmc.setProperty('HamamatsuHam_DCAM', 'OUTPUT TRIGGER KIND[1]', 'EXPOSURE')
mmc.setProperty('HamamatsuHam_DCAM', 'TRIGGER ACTIVE', 'LEVEL')
mmc.setProperty('HamamatsuHam_DCAM', 'TRIGGER GLOBAL EXPOSURE', 'GLOBAL RESET')
#mmc.setProperty('HamamatsuHam_DCAM', 'TRIGGER TIMES', '1500')

mmc.setProperty('Arduino-Switch', 'Blank On', 'High')

mmc.setAutoShutter(True)    
ZStage = mmc.getFocusDevice()
mmc.setPosition(ZStage, 50)  # moving piezo to its middle position
mmc.setExposure(50)

import numpy as np
from tifffile import TiffWriter
import queue


MAX_FILE_SIZE = 2 * 1024**3

'''class TiffWriterManager(threading.Thread):
    def __init__(self,  frame_queue, save_filename):
        self.save_filename = save_filename
        self.queue = frame_queue
        self.file_index = 0
        self.bytes_written = 0
        self.writer = None
        #self._open_new_writer()

    def _open_new_writer(self):
        if self.writer:
            self.writer.close()
        fname = f"{self.save_filename}_{self.file_index:03d}.tif"
        self.writer = TiffWriter(fname, bigtiff=True)
        print(f"[Writer] Opened new file: {fname}")
        self.bytes_written = 0
        self.file_index += 1

    def write(self, image: np.ndarray):
        if image is None:
            return
        image_bytes = image.nbytes
        if self.bytes_written + image_bytes > MAX_FILE_SIZE:
            self._open_new_writer()
        self.writer.write(image, contiguous=True)
        self.bytes_written += image_bytes

    def close(self):
        if self.writer:
            self.writer.close()
            print("[Writer] Closed.")
            self.writer = None
            
def writer_thread_func(image_queue, stop_event, save_filename):
    manager = TiffWriterThread(save_filename)
    while not stop_event.is_set() or not image_queue.empty():
        try:
            frame, metadata = image_queue.get(timeout=0.05)
            manager.write(frame)
        except queue.Empty:
            continue
    manager.close()'''

            
class TiffWriterThread(threading.Thread):
    def __init__(self, frame_queue, base_filename):
        super().__init__()
        self.queue = frame_queue
        self.base_filename = base_filename
        self.file_index = 0
        self.current_writer = None
        self.current_file = None
        self.current_size = 0
        self.stop_flag = False
        #print('started at least')

    def new_file(self):
        if self.current_writer:
            self.current_writer.close()

        filename = file_name_check(f"{self.base_filename}_{self.file_index:02d}.tif")
        self.current_file = filename
        self.current_writer = TiffWriter(filename, bigtiff=False)
        self.current_size = 0
        self.file_index += 1
        print(f"Opened new file: {filename}")

    def run(self):
        self.new_file()
        print('run')
        while not self.stop_flag or not self.queue.empty():
            if self.stop_flag and self.queue.empty():
                break
            try:    
                try:
                    frame = self.queue.get(timeout=0.05)
                    '''if frame is None:
                        print('break')
                        break'''
                    #print('frame received')
                    #print(f"Frame info: type={type(frame)}, dtype={getattr(frame, 'dtype', None)}, shape={getattr(frame, 'shape', None)}")
                except queue.Empty:
                    continue  
                # Write frame
                self.current_writer.write(frame, contiguous=True)
                #print('frame written')
                # Track size
                self.current_size += frame.nbytes
                if self.current_size > MAX_FILE_SIZE:
                    self.new_file()
           
            except Exception as e:
                print(f"❌ Writer thread crashed: {e}")
                #self.log_event(f"Writer crashed with error: {e}")
           
        if self.current_writer:
            self.current_writer.close()
        print("Writer thread finished.")

    def stop(self):
        self.stop_flag = True


def file_name_check(path):
    filename, extension = os.path.splitext(path)
    counter = 2
    while os.path.exists(path):
        path = filename + "_" +str(counter)+""+ extension
        counter += 1
    return path

def norm_data(data): # normalize data to have mean=0 and standard_deviation=1
    mean_data=np.mean(data)
    std_data=np.std(data, ddof=1)
    #return (data-mean_data)/(std_data*np.sqrt(data.size-1))
    return (data-mean_data)/(std_data)

def ncc(data0, data1): #    normalized cross-correlation coefficient between two data sets
    return (1.0/(data0.size-1)) * np.sum(norm_data(data0)*norm_data(data1))


class LiveThread(QThread):
    frame_received = pyqtSignal(np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = False

    def run(self):
        #pattern = (["4","8","16","32"])
        #mmc.setProperty('Arduino-Switch',  'Sequence', "On")
        #mmc.loadPropertySequence("Arduino-Switch", "State", pattern)
        with mmc_lock:
            mmc.startContinuousSequenceAcquisition()
            mmc.setProperty('Arduino-Shutter',  'OnOff', '1')
        #mmc.startPropertySequence("Arduino-Switch", "State")
        
        self.running = True
        try:
            while self.running:
                if mmc.getRemainingImageCount() > 0:
                    image = mmc.popNextImage()
                    #im.append(rgb32)
                    self.frame_received.emit(image)
                #self.msleep(30)  # ~33 FPS
        finally:
            with mmc_lock:
                if mmc.isSequenceRunning():
                    mmc.stopSequenceAcquisition()
                mmc.setProperty('Arduino-Shutter', 'OnOff', '0')
            #mmc.stopPropertySequence("Arduino-Switch", "State")

    def stop(self):
        self.running = False
        #self.wait()
        

class GetReference(QThread):
    frame_received = pyqtSignal(np.ndarray)
    logs_signal = pyqtSignal(str)
    ref_recorded = pyqtSignal(np.ndarray)
    def __init__(self,  main_window):
        super().__init__()
        self.main_window = main_window
        self.running = False
        #self.pattern = pattern
        #self.ref_image = ref_image

    def run(self):
        pattern =  self.main_window.pattern
        self.logs_signal.emit("Acquiring reference, please wait ! and the pattern is: " + str(pattern) )
        savefolder = self.main_window.savepath
        
        if not os.path.exists(savefolder):
            self.logs_signal.emit("Save folder does not exist" )
            return
        filepath = file_name_check(os.path.join(savefolder, "ref_"+self.main_window.filename+".tiff"))
        self.logs_signal.emit("save path "+filepath )
        with mmc_lock:
            if mmc.isSequenceRunning() == True:
                mmc.stopSequenceAcquisition() 
            mmc.setProperty('Arduino-Switch',  'Sequence', "On")
            mmc.loadPropertySequence("Arduino-Switch", "State", pattern)  
        start = -45
        stop = 45
        step = 10
        pos0 = mmc.getPosition(ZStage)
        images = []
        zpos = start
        mmc.setPosition(ZStage, pos0+zpos)
        mmc.startContinuousSequenceAcquisition()
        mmc.startPropertySequence("Arduino-Switch", "State")
        mmc.setProperty('Arduino-Shutter',  'OnOff', '1')
        self.running = True
        try:
            while self.running:
                time.sleep(0.001)
                while zpos <=stop:
                    with mmc_lock:
                        mmc.setPosition(ZStage, pos0+zpos)
                        if mmc.getRemainingImageCount() > 0:
                            image = mmc.popNextImage()
                        images.append(image)
                        self.frame_received.emit(image)
                        if (len(images) % (len(pattern)*2) == 0)== True: # 2 - number of images to avarage
                            zpos = zpos +step
                            self.logs_signal.emit("zpoz =  "+str(zpos)+"len Images"+str(len(images)) )
                self.running = False           
        except Exception as e:
            self.logs_signal.emit(f"LiveThread top-level error: {e}")
        finally:
            with mmc_lock:
                mmc.setPosition(ZStage, pos0)
                if mmc.isSequenceRunning():
                    mmc.stopSequenceAcquisition()
                mmc.setProperty('Arduino-Shutter', 'OnOff', '0')
            ref_image=([np.mean(np.asarray(images)[i::len(pattern)], axis=0) for i in range(0,len(pattern))])
            imwrite(filepath, np.asarray(ref_image).astype(np.float32))
            self.ref_recorded.emit(np.asarray(ref_image))
            self.logs_signal.emit("Reference acquired and saved" )
            mmc.stopPropertySequence("Arduino-Switch", "State")
                
    def stop(self):
        self.running = False


class LivehDPCThread(QThread):
    frame_received = pyqtSignal(np.ndarray)
    logs_signal = pyqtSignal(str)
    
    def __init__(self,   main_window):
        super().__init__()
        self.running = False
        self.main_window = main_window
        #self.pattern = pattern
        self.ref_image_ =  None
        
    @pyqtSlot(object)  # Accept list of np.ndarray
    def set_reference_image(self, ref_image):
        self.ref_image_ = ref_image.copy()  # Thread-safe copy

    def hDPC(self,sp1, sp2):   ## function to calculate differential phase contrast / gradient images
        Idpc = (sp1-sp2)/(sp1+sp2)
        mean = Idpc.mean()
        std = Idpc.std()
        Idpc_c = np.clip(Idpc, mean-4*std, mean+4*std)
        return Idpc_c

    def run(self):
        pattern =  self.main_window.pattern
        #self.main_window.image_view.image = None
        self.logs_signal.emit("Starting live view with "+str(len(pattern))+"LEDs illumination" )
        with mmc_lock: 
            if mmc.isSequenceRunning() ==True:
                mmc.stopSequenceAcquisition()   
            mmc.setProperty('Arduino-Switch',  'Sequence', "On")
            mmc.setProperty('Arduino-Shutter',  'OnOff', '1')
            mmc.loadPropertySequence("Arduino-Switch", "State", pattern)
            mmc.startPropertySequence("Arduino-Switch", "State")
            mmc.startContinuousSequenceAcquisition()
        self.running = True
        two = []
        try:
            while self.running:
                time.sleep(0.001)
                with mmc_lock:
                    if mmc.getRemainingImageCount() > 0:
                        image = mmc.popNextImage()
                        two.append(image)
                        if len(pattern) ==4 and len(two) == len(pattern):
                            #self.logs_signal.emit(str(np.asarray(two).shape)+str( np.asarray(self.ref_image_).shape))
                            result = self.hDPC(np.asarray(two)[0]/np.asarray(self.ref_image_)[0], np.asarray(two)[1]/np.asarray(self.ref_image_)[1])
                            self.frame_received.emit(result)
                            two=[]
                        if len(pattern) == 2 and len(two) == len(pattern):
                            #self.logs_signal.emit(str(np.asarray(two).shape)+str( np.asarray(self.ref_image_).shape))
                            result = (np.asarray(two)[0]/np.asarray(self.ref_image_)[0] - np.asarray(two)[1]/np.asarray(self.ref_image_)[1])/(np.asarray(two)[0]/np.asarray(self.ref_image_)[0] + np.asarray(two)[1]/np.asarray(self.ref_image_)[1])
                            self.frame_received.emit(result)
                            two=[]
            #mmc.stopSequenceAcquisition() 
        except Exception as e:
            self.logs_signal.emit(f"LiveThread top-level error: {e}")
        finally:
            with mmc_lock:
                if mmc.isSequenceRunning():
                    mmc.stopSequenceAcquisition()
                mmc.setProperty('Arduino-Shutter', 'OnOff', '0')
                mmc.stopPropertySequence("Arduino-Switch", "State")
            self.running = False
                         
    def stop(self):
        self.running = False
        self.wait()
        

class AcuqisitionThread(QThread):
    frame_received = pyqtSignal(np.ndarray)
    logs_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()
    def __init__(self, nIm, frame_queue, stop_event, main_window):
        super().__init__()
        self.main_window = main_window
        #self.pattern = pattern
        self.running = False
        self.nIm = nIm
        self.frame_queue = frame_queue
        self.stop_event = stop_event
        
    def set_reference_image(self, ref_image):
        self.ref_image_ = ref_image.copy()

    def run(self):
        self.main_window.image_view.image = None
        pattern =  self.main_window.pattern
        self.logs_signal.emit("Acquiring images: " + str(self.nIm) )
        savefolder = self.main_window.savepath
        if not os.path.exists(savefolder):
            self.logs_signal.emit("Save folder does not exist" )
            return
        filepath = os.path.join(self.main_window.savepath, "acq_"+self.main_window.filename+".tiff")
        #self.tiff_writer = TiffWriter(filepath, bigtiff=True)
        with mmc_lock:
            if mmc.isSequenceRunning() == True:
                mmc.stopSequenceAcquisition() 
            mmc.setProperty('Arduino-Switch',  'Sequence', "On")
            mmc.setProperty('Arduino-Shutter',  'OnOff', '1')
            mmc.loadPropertySequence("Arduino-Switch", "State", pattern)
            mmc.startPropertySequence("Arduino-Switch", "State")
            mmc.startContinuousSequenceAcquisition()
        self.running = True
        image_count = 0
        startTime = time.time()
        timeStamp=[]
        #two = []
        try:
            while self.running:
                time.sleep(0.001)
                with mmc_lock:
                    if mmc.getRemainingImageCount() > 0:
                        image = mmc.popNextImage()
                        metadata = {'timestamp': time.time()}
                        image_count = image_count+1
                        #two.append(image)
                        #if len(two) == len(pattern):
                            #self.logs_signal.emit(str(np.asarray(two).shape)+str( np.asarray(self.ref_image_).shape))
                            #result = (np.asarray(two)[0] - np.asarray(two)[1])/(np.asarray(two)[0] + np.asarray(two)[1])
                            #self.frame_received.emit(result)
                            #two=[]
                        timeStamp.append(time.time()-startTime)
                        self.frame_received.emit(image)
                        try:
                            self.frame_queue.put(image.astype(np.float16))  
                        except queue.Full:
                            self.logs_signal.emit("[WARN] Queue full — frame dropped")
                        if image_count == self.nIm:
                            self.running = False
        except Exception as e:
            self.logs_signal.emit(f"LiveThread top-level error: {e}")
        finally:
            with mmc_lock: 
                if mmc.isSequenceRunning():
                    mmc.stopSequenceAcquisition()
                mmc.setProperty('Arduino-Shutter', 'OnOff', '0')
                mmc.stopPropertySequence("Arduino-Switch", "State")
            with open(os.path.join(savefolder + '\\METADATA.txt'), 'a') as file:
                file.write(' ___ image index:  %s \n timestamp: %s ' %(image_count, timeStamp))
               
            #self.tiff_writer.close()
            self.logs_signal.emit("Acquisition finished, time: "+str(time.time()-startTime) )
            self.running = False
            #self.frame_queue.put(None)
            self.finished_signal.emit()
             
    def stop(self):
        self.running = False
        self.stop_event.set()
 
def get_image(pattern, av=1):   ## halping function, acquires 4 images with 4 leds
    im=[]
    with mmc_lock:
        mmc.loadPropertySequence("Arduino-Switch", "State", pattern)
        mmc.startSequenceAcquisition(8*av, 0, True)
        mmc.startPropertySequence("Arduino-Switch", "State")
    #mmc.startPropertySequence("Arduino-Switch", "State")
    skipped = 0
    skip_count = 4
    while mmc.isSequenceRunning():
        if mmc.getRemainingImageCount() != 0:
            image = mmc.popNextImage()
            if skipped< skip_count:
               skipped+=1
               #self.logs_signal.emit("iamge skipped")
               continue
            im.append(image)
    return im 

startt = -1.5
stopp = 1.5
stepp = 0.05

class Record_Zstack(QThread):
    frame_received = pyqtSignal(np.ndarray)
    logs_signal = pyqtSignal(str)
    refStack_recorded = pyqtSignal(np.ndarray)
    
    def __init__(self,  main_window):
        super().__init__()
        self.running = False
        self.main_window = main_window
        #self.pattern = pattern
        self.ref_image_ =  None
        
    @pyqtSlot(object)  # Accept list of np.ndarray
    def set_reference_image(self, ref_image):
        self.ref_image_ = ref_image.copy() 
    def hDPC(self,sp1, sp2):   ## function to calculate differential phase contrast / gradient images
        Idpc = (sp1-sp2)/(sp1+sp2)
        mean = Idpc.mean()
        std = Idpc.std()
        Idpc_c = np.clip(Idpc, mean-4*std, mean+4*std)
        return Idpc_c
        
    def run(self):
        pattern =  self.main_window.pattern
        self.main_window.reset_autolevels = True # to make image viewer restart the brightnes adjustment
        savefolder = self.main_window.savepath
        if not os.path.exists(savefolder):
            self.logs_signal.emit("Save folder does not exist" )
            return
        if self.ref_image_ is None:
            list_of_files = glob.glob(os.path.normpath(savefolder+ '\\*'))
            refpath = os.path.normpath(max([file for file in list_of_files if 'ref' in file], key=os.path.getmtime))
            self.ref_image_ =  imread(refpath)  
            self.logs_signal.emit("aploading  ref done")
        filepath = file_name_check(os.path.join(self.main_window.savepath, "zStack"+self.main_window.filename+".tiff"))
        self.tiff_writer = TiffWriter(filepath, bigtiff=True)
        self.logs_signal.emit("Starting reference zStack acquisition with "+str(len(pattern))+"LEDs illumination" )
        with mmc_lock:
            if mmc.isSequenceRunning() ==True:
                mmc.stopSequenceAcquisition()   
        start = startt#-0.75
        stop = stopp#0.75
        step = stepp #0.05
        pos0 = mmc.getPosition(ZStage)
        images = []
        posRead = []
        zpos = start
        TimeStart = time.time()
        stack = []
        with mmc_lock:
            mmc.setProperty('Arduino-Switch',  'Sequence', "On")
            mmc.loadPropertySequence("Arduino-Switch", "State", pattern)
            mmc.startPropertySequence("Arduino-Switch", "State")
            mmc.startContinuousSequenceAcquisition()
            mmc.setProperty('Arduino-Shutter',  'OnOff', '1')
        skipped = 0
        skip_count = 4
        self.running = True
        try:
            while self.running:
                time.sleep(0.001)
                while zpos <=stop:
                    with mmc_lock: 
                        mmc.setPosition(ZStage, pos0+zpos)
                        mmc.waitForDevice(ZStage)
                        if mmc.getRemainingImageCount() > 0:
                            image = mmc.popNextImage()
                            if skipped< skip_count:
                               skipped+=1
                               self.logs_signal.emit("iamge skipped")
                               continue
                            images.append(image)
                            stack.append(image)
                            #metadata={"timestamp": time.time()}
                            self.tiff_writer.write(image.astype(np.float16), contiguous=True)
                            self.frame_received.emit(image)
                            if (len(images) % (len(pattern)*1) == 0)== True: # 2 - number of images to avarage
                                zpos = zpos +step
                                self.logs_signal.emit("zpoz =  "+str(zpos)+", len Images "+str(len(images)))
                                posRead.append(mmc.getPosition(ZStage))
                                images = []
                self.running = False           
        except Exception as e:
            self.logs_signal.emit(f"zStack top-level error: {e}")
        finally:
            self.logs_signal.emit("z Stack in range -1.5:1.5 with step 50nm done in "+str(time.time()-TimeStart) )
            mmc.setPosition(ZStage, pos0)
            self.refStack_recorded.emit(np.asarray(stack))
            if mmc.isSequenceRunning():
                mmc.stopSequenceAcquisition()
            mmc.setProperty('Arduino-Shutter', 'OnOff', '0')
            self.tiff_writer.close()
            zStack = np.asarray(stack).reshape(int(len(np.asarray(stack))/4), 4, 2048, 2048)
            self.logs_signal.emit("getting hDPC stack, shape = "+str(zStack.shape) )
            Idpc1 = self.hDPC(np.asarray(zStack)[:,3]/np.asarray(self.ref_image_)[3], np.asarray(zStack)[:,0]/np.asarray(self.ref_image_)[0]) 
            Idpc2 = self.hDPC(np.asarray(zStack)[:,2]/np.asarray(self.ref_image_)[2], np.asarray(zStack)[:,1]/np.asarray(self.ref_image_)[1]) 
            z_Stack=Idpc2+Idpc1
            imwrite(file_name_check(os.path.join(self.main_window.savepath, "z_hDPCStack"+self.main_window.filename+".tiff")), np.asarray(z_Stack).astype(np.float32))
            mmc.stopPropertySequence("Arduino-Switch", "State")
            np.savetxt(file_name_check(os.path.join(self.main_window.savepath, "zStack_posRead"+self.main_window.filename+".txt")), posRead)
            self.logs_signal.emit("Reference zStack acquired and saved" )
            self.running = False
                         
    def stop(self):
        self.running = False
        #self.wait()
        
class ACQautofocus(QThread):    ## The function to perform acquisition with active autofocusing
    frame_received = pyqtSignal(np.ndarray)
    logs_signal = pyqtSignal(str)
    data1_ready =  pyqtSignal(float)
    data2_ready =  pyqtSignal(np.ndarray, np.ndarray)
    
    def __init__(self, nIm, freq, ref_image, ref_zStack, main_window):
        super().__init__()
        self.running = False
        self.main_window = main_window
        #self.pattern = pattern
        self.ref_image_ =  None
        self.zStack = None
        self.nIm = nIm
        self.upload_ref = False
        self.freq = freq
        self.ref_image_ = ref_image
        self.zStack = ref_zStack
        
        self.fit_degree = 10

    @pyqtSlot(object)  # Accept list of np.ndarray
    def set_reference_image(self, ref_image):
        self.ref_image_ = ref_image.copy()  # Thread-safe copy
        
    @pyqtSlot(int)
    def update_degree_value(self, new_fit_degree_value):
        """Called from main GUI whenever user updates input field."""
        self.fit_degree = new_fit_degree_value
        self.logs_signal.emit(f"Updated value in acquisition thread: {new_fit_degree_value}")
    
    @pyqtSlot(object) 
    def set_ref_zStack(self, zStack):
        self.zStack = zStack.copy()

    def hDPC(self,sp1, sp2):   ## function to calculate differential phase contrast / gradient images
        Idpc = (sp1-sp2)/(sp1+sp2)
        mean = Idpc.mean()
        std = Idpc.std()
        Idpc_c = np.clip(Idpc, mean-4*std, mean+4*std)
        return Idpc_c

    def run(self):
        pattern =  self.main_window.pattern
        self.main_window.image_view.image = None
        savefolder = self.main_window.savepath
        if not os.path.exists(savefolder):
            self.logs_signal.emit("Save folder does not exist" )
            return
        if self.ref_image_ is None:
            list_of_files = glob.glob(os.path.normpath(savefolder+ '\\*'))
            refpath = os.path.normpath(max([file for file in list_of_files if 'ref' in file], key=os.path.getmtime))
            self.ref_image_ =  imread(refpath)  
            self.logs_signal.emit("aploading  ref done")
        if self.zStack is None:
            list_of_files = glob.glob(os.path.normpath(savefolder+ '\\*.tiff'))
            zStackpath = os.path.normpath(max([file for file in list_of_files if 'zStack' in file], key=os.path.getmtime))
            self.logs_signal.emit("aploading  zStack" + str(zStackpath))
            self.zStack =  imread(zStackpath) 
        filepath = file_name_check(os.path.join(self.main_window.savepath, "AF_Stack"+self.main_window.filename+".tiff"))
        self.tiff_writer = TiffWriter(filepath, bigtiff=True)
        self.logs_signal.emit("Starting acquisition with AF "+str(len(pattern))+"LEDs illumination" )
        self.logs_signal.emit("length of zStack = "+str(len(self.zStack))+ " the nIm = "+ str(self.nIm)+ " freq = "+ str( self.freq))
        self.zStack = np.asarray(self.zStack).reshape(int(len(np.asarray(self.zStack))/4), 4, 2048, 2048)
        self.logs_signal.emit("calculation of hDPCs")
        Idpc1 = self.hDPC(np.asarray(self.zStack)[:,3]/np.asarray(self.ref_image_)[3], np.asarray(self.zStack)[:,0]/np.asarray(self.ref_image_)[0]) 
        Idpc2 = self.hDPC(np.asarray(self.zStack)[:,2]/np.asarray(self.ref_image_)[2], np.asarray(self.zStack)[:,1]/np.asarray(self.ref_image_)[1]) 
        z_Stack=Idpc2+Idpc1
        self.logs_signal.emit("calculation done")
        #z_Stack =  self.hDPC(np.asarray(self.zStack)[:,0]/np.asarray(self.ref_image_)[0], np.asarray(self.zStack)[:,1]/np.asarray(self.ref_image_)[1])
        positionList = (np.arange(startt, stopp, stepp))#(np.arange(-0.75, 0.75, 0.05))
        x = positionList#-positionList[int(len(positionList)/2)]
        xx = np.arange(x.min(), x.max(), 0.001)
        self.logs_signal.emit("zstack shape before/ after "+str(np.asarray(self.zStack).shape)+"\\"+str(np.asarray(z_Stack).shape))
        frame = int(len(z_Stack)/2)
        sift_ocl = sift.SiftPlan(template=z_Stack[frame, 600:1300, 600:1300], devicetype="GPU")
        keypoints = sift_ocl(z_Stack[frame, 600:1300, 600:1300])
        mp = sift.MatchPlan()
        self.logs_signal.emit("features stuff")
        images = []
        drft = []
        micPos=[]
        #degree = 12
        Im = []
        i=0
        self.logs_signal.emit("starting")
        if mmc.isSequenceRunning() == True:
            mmc.stopSequenceAcquisition()   
        with mmc_lock:
            mmc.setProperty('Arduino-Switch',  'Sequence', "On")
            mmc.setProperty('Arduino-Shutter',  'OnOff', '1')
        imm=[]
        self.running = True  
        image_count = 0
        skipped=0
        skip_count= len(pattern)
        startTime = time.time()
        timeStamp=[]
        try:
            while self.running:
                time.sleep(0.001)
                startTime = time.time()
                #image = mmc.getImage(mmc.snapImage())#mmc.popNextImage()
                try:
                    image = get_image(pattern)
                except RuntimeError:
                    image = get_image(pattern)
                    pass
                images.append(image)
                #print('appended')
                if len(images) % (self.freq)==0:
                    self.logs_signal.emit("go...")
                    timeStamp.append(time.time()-startTime)
                    images = np.asarray(images).reshape((len(images*4),2048, 2048))
                    self.logs_signal.emit(str(np.asarray(images).shape)+str( np.asarray(self.ref_image_).shape))
                    Idpc1im  = self.hDPC(np.asarray(images)[-4]/np.asarray(self.ref_image_)[3], np.asarray(images)[-3]/np.asarray(self.ref_image_)[0])
                    Idpc2im  = self.hDPC(np.asarray(images)[-1]/np.asarray(self.ref_image_)[2], np.asarray(images)[-2]/np.asarray(self.ref_image_)[1])
                    Im = Idpc2im+Idpc1im
                    #print('here 1.2')
                    self.tiff_writer.write(Im.astype(np.float16), contiguous=True)
                    self.frame_received.emit(Idpc2+Idpc1)
                    pos = mmc.getPosition(ZStage)
                    corrcoef = []
                    if not self.running:
                        break
                    try: 
                        #print('here 1.3')
                        im_keypoints = sift_ocl(np.asarray(Im[600:1300, 600:1300]).astype(np.float32))
                        match = mp(keypoints, im_keypoints)
                        sa = sift.LinearAlign(z_Stack[frame,600:1300, 600:1300], devicetype="GPU")   
                        im = sa.align(np.asarray(Im[600:1300, 600:1300]).astype(np.float32),  shift_only=True)
                    except:
                        self.logs_signal.emit("Something went wrong with xy drift corretion, saving the data...")
                        np.savetxt(file_name_check(os.path.join(self.main_window.savepath, "detectDrift_fail"+self.main_window.filename+".txt")), pd.concat([pd.DataFrame(drft), pd.DataFrame(micPos)], axis=1))
                    if im is None:
                        self.logs_signal.emit(" xy-drfit correction didn't work")
                    #print('here 1.4')    
                    for f in range(0, len(z_Stack)):
                        #print("loop startts")
                        try:
                            corrcoef.append(ncc(im, z_Stack[f,600:1300, 600:1300]))
                            #print("corrcorf thing, the sizes are: " +str(im.shape)+str(z_Stack.shape))
                        except:
                            np.savetxt(file_name_check(os.path.join(self.main_window.savepath, 'detectedDrift.txt')), pd.concat([pd.DataFrame(drft), pd.DataFrame(micPos)], axis=1))
                            #with open(os.path.join(self.main_window.savepath, + '\\METADATA.txt', 'a') as file:
                             #   file.write('Aqqusition with autofocus recorded FAIL correlation at %s\n N of images %s \n frequency:%s\n Exposure: %s\n  degree: %s\n ' %(datetime.datetime.now(), nIm, frequency, exposure, degree))
                            print('xy correction failed, data was saved')
                            break
                    image_count = image_count+1
                    if image_count == self.nIm:
                        self.running = False
                    images=[]
                    y = np.array(corrcoef)
                    self.logs_signal.emit("len y ="+str(len(y))+" len x = "+str(len(x)))
                    model2 = np.poly1d(np.polyfit(x, y, self.fit_degree))
                    drft.append(xx[where(model2(xx) == model2(xx).max())])
                    #print("before heer 2"+str(drft[-1][0]))
                    self.data1_ready.emit(drft[-1][0])
                    #print('here 2')
                    self.data2_ready.emit(np.array(model2(xx)).flatten(), y.flatten())
                    #print('here 3')
                    micPos.append(mmc.getPosition(ZStage))
                    self.logs_signal.emit("The drift detected was: "+str(drft[-1])+" it took: "+str(time.time()-startTime))
                    if (np.abs(drft[-1]) < 0.010):
                        self.logs_signal.emit('drift was smaller then 10 nm, no correction is happening')
                        continue 
                    mmc.setPosition(ZStage, pos-drft[-1][0])
                    mmc.waitForDevice(ZStage)
                    skipped=0     
            self.running = False  
        except Exception as e:
            self.logs_signal.emit(f"AcqAF thread top-level error: {e}")            
        finally:
            if mmc.isSequenceRunning():
                mmc.stopSequenceAcquisition()
            mmc.setProperty('Arduino-Shutter', 'OnOff', '0')
            mmc.stopPropertySequence("Arduino-Switch", "State")
            #np.savetxt(file_name_check(os.path.join(self.main_window.savepath, "zStack_posRead"+self.main_window.filename+".tiff")), posRead)
            np.savetxt(file_name_check(os.path.join(self.main_window.savepath, 'detectedDrift.txt')), pd.concat([pd.DataFrame(drft), pd.DataFrame(micPos)], axis=1))
            self.tiff_writer.close()
            self.logs_signal.emit("AF Stack acquired and saved" )
            self.running = False
            with open(os.path.join(savefolder + '\\METADATA.txt'), 'a') as file:
                file.write(' \n___ acq AF image index:  %s \n timestamp: %s ' %(image_count, timeStamp))
                     
    def stop(self):
        self.running = False
        #self.wait()


class PlotWindow(QWidget):
    def __init__(self):
        super().__init__()
        uic.loadUi("D:/Data_Programs/CODE/LedControl/plotWidget.ui", self)
        
        self.plot1 = pg.PlotWidget(title="Detected Drift")
        #self.setupUi(self) 
     
        layout = QVBoxLayout()
        layout.addWidget(self.plot1)
        self.graph1.setLayout(layout)  # your QWidget from .ui
    
        self.plot2 = pg.PlotWidget(title="CC curves and the model")
        
        layout2 = QVBoxLayout()
        layout2.addWidget(self.plot2)
        self.graph2.setLayout(layout2)  # your QWidget from .ui

        self.data1 = []
        self.curve1 = self.plot1.plot(pen='y')
        
        #self.data2 = []
        self.data_model = []
        self.data_ccc = []
        self.curve_model = self.plot2.plot(pen='y', name = "Model")
        self.curve_ccc = self.plot2.plot(pen='c', name = "CCC")
        
        self.plot2.addLegend()

    def update_plot1(self, value):
        self.data1.append(value)
        self.curve1.setData(self.data1)
    
    def update_plot2(self, value1, value2):
        #self.data2 = value
        x_model = np.linspace(0, 60, len(self.data_model))  
        self.data_model = value1
        self.data_ccc = value2
        self.curve_model.setData(x_model, self.data_model)
        self.curve_ccc.setData(self.data_ccc)
        
    def reset(self):
        """Clear plots and reset stored data."""
        self.data1.clear()
        self.data_model.clear()
        self.data_ccc.clear()
        self.curve1.clear()
        self.curve_model.clear()
        self.curve_ccc.clear()

        

###   ________________________________________________________________________________________________________main window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi('D:/Data_Programs/CODE/LedControl/LEDdisplay2.3.ui', self)
        #uic.loadUi("D:/!Hanna/!PythonCode/LedControl/LEDdisplay2.3.ui", self)  # Replace with your .ui file path

        self.plot_window = PlotWindow()

        self.image_view = ImageView()
        layout = QVBoxLayout()
        layout.addWidget(self.image_view)
        self.WidgetWindow.setLayout(layout)  # your QWidget from .ui

        self.pos_info_label = QLabel("X | Y | I")
        self.statusBar().addWidget(self.pos_info_label)
        self.roi = pg.ROI([100, 100], [200, 200], pen='r')
        self.roi.addScaleHandle([1, 1], [0, 0])
        self.roi.addScaleHandle([0, 0], [1, 1])
        self.roi.setZValue(10)  # Make sure ROI stays on top
        #self.roi.setBounds(QRectF(0, 0, 2048, 2048))
        
        # Add it to the same ViewBox as the image
        self.image_view.addItem(self.roi)
        
        self.set_roi.clicked.connect(self.roi_changed)
        self.reset_roi.clicked.connect(self.roi_reset)
        #self.roi.sigRegionChanged.connect(self.roi_changed)
        
        self.frame_queue = queue.Queue()#queue.Queue(maxsize=100)
        
        self.stop_event = threading.Event()
        self.writer_thread = None
        self.acq_thread = None

        self.image_view.getView().scene().sigMouseMoved.connect(self.mouse_moved)
        
        self.current_frame = None
        self.pattern = ""
        self.ref_image = None
        self.ref_zStack = None
        self.savepath = ""
        self.filename = ""
        
        self.illumination_pattern_change('4 LED')

        self.StartLive.setCheckable(True)
        self.StartLive.clicked.connect(self.toggle_live)
        
        self.live_thread = LiveThread()
        self.live_thread.frame_received.connect(self.update_frame)
        
        self.GetRef.setCheckable(True)
        self.GetRef.clicked.connect(self.toggle_ref)
        self.refacquisition_thread = GetReference(main_window=self)
        self.refacquisition_thread.logs_signal.connect(self.append_logs)
        self.refacquisition_thread.frame_received.connect(self.update_frame)
        self.refacquisition_thread.ref_recorded.connect(self.store_reference_image)
        
        self.browsePath.clicked.connect(self.browse_path)
        
        self.StartAqcuisition.setCheckable(True)
        self.StartAqcuisition.clicked.connect(self.toggle_acquisition)
        
        #self.acquisition_thread = AcuqisitionThread( self.Nimages.value(), main_window=self)
       # self.acquisition_thread.logs_signal.connect(self.append_logs)
        #self.acquisition_thread.frame_received.connect(self.update_frame)
        #  live hDPC        
        self.StartLiveHDPC.setCheckable(True)
        self.StartLiveHDPC.clicked.connect(self.toggle_liveHDPC)
        self.liveHDPC_thread = LivehDPCThread( main_window=self)
        self.liveHDPC_thread.logs_signal.connect(self.append_logs)
        self.liveHDPC_thread.frame_received.connect(self.update_frame)
        
        # Acquisition of ref zStack
        self.RefzStack.setCheckable(True)
        self.RefzStack.clicked.connect(self.toggle_refzStack)
        self.ref_zstack = Record_Zstack( main_window=self)
        self.ref_zstack.logs_signal.connect(self.append_logs)
        self.ref_zstack.frame_received.connect(self.update_frame)
        self.ref_zstack.refStack_recorded.connect(self.store_refzStack)
        
        # Acquisition with AF
        self.AcqAF.setCheckable(True)
        self.AcqAF.clicked.connect(self.toggle_acqAF)    
        
        self.save_path.textChanged.connect(self.set_save_path)
        self.fileName.textChanged.connect(self.set_file_name) 
        
        #self.degree.textChamged.connect(self.set_degree)
        
        self.UpdateLevels.clicked.connect(self.reset_autolevels)
        self.InteruptButton.clicked.connect(self.interrupt_all_threads)
        
        self.ExposureTime.valueChanged.connect(self.set_exposure)
        self.IlluminationLED.currentTextChanged.connect(self.illumination_change)
        self.IlluminationLED_pattern.currentTextChanged.connect(self.illumination_pattern_change)
        self.piezoPos.valueChanged.connect(self.set_piezo_position)

    def roi_changed(self):
        pos = self.roi.pos()
        size = self.roi.size()
        x, y = int(pos.x()), int(pos.y())
        w, h = int(size.x()), int(size.y())
        if self.live_thread:
            self.live_thread.stop()
        mmc.setROI(y,x,h,w) 
        print(f"ROI: x={x}, y={y}, w={w}, h={h}")
        
    def roi_reset(self):
        if self.live_thread:
            self.live_thread.stop()
        mmc.setROI(0,0,2048,2048)

    def set_piezo_position(self, value):
        mmc.setPosition(ZStage, mmc.getPosition(ZStage)+value)
    
    @pyqtSlot(object)
    def store_reference_image(self, ref_image):
        self.ref_image = [img.copy() for img in ref_image]  # Store copy
        self.liveHDPC_thread.set_reference_image(self.ref_image)
        #self.StartAqcuisition.set_reference_image(self.ref_image)
        self.ref_zstack.set_reference_image(self.ref_image)
        
        #self.Acq_AF.set_reference_image(self.ref_image)
        self.append_logs("Reference image stored and passed to HDPC thread.")
        
    def store_refzStack(self, zStack):
        self.ref_zStack = [img.copy() for img in zStack]
        #self.Acq_AF.set_ref_zStack(self.ref_zStack)
        self.append_logs("Reference image stored and passed to HDPC thread.")
    
    def browse_path(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file')
        self.save_path.setText(fname[0])    

    def set_save_path(self, text):  
        self.savepath = text#self.save_path.text()
        
    def set_file_name(self, text):
        self.filename = text #self.fileName.text()

    def toggle_live(self, checked):
        if checked:
            self.live_thread.start()
            self.StartLive.setText("Stop Live")
        else:
            self.live_thread.stop()
            self.StartLive.setText("Start Live")

    def toggle_acquisition(self, checked):
        if checked:
            self.stop_event.clear()
            n_images = self.Nimages.value()
            savefolder = self.savepath
            save_filename = os.path.join(savefolder, "acq_")
            #self.writer_thread = threading.Thread(target=writer_thread_func,args=(self.image_queue, self.stop_event, save_filename),daemon=True )
            self.writer = TiffWriterThread(self.frame_queue, save_filename)
            
            self.acquisition_thread = AcuqisitionThread(n_images, self.frame_queue, self.stop_event, main_window=self)
            self.acquisition_thread.logs_signal.connect(self.append_logs)
            self.acquisition_thread.frame_received.connect(self.update_frame)
            self.acquisition_thread.finished_signal.connect(self.stop_acquisition)
            
            #self.writer_thread.start()
            self.writer.start()
            self.acquisition_thread.start()
            self.StartAqcuisition.setText("Stop Acquisition")
        else:
            #self.stop_event.set()
            self.acquisition_thread.stop()
            if self.writer:
                self.writer.stop()
                self.writer.join()
            self.StartAqcuisition.setText("Start Acquisition")
            
    def stop_acquisition(self):
        self.acquisition_thread.stop()
        if self.writer:
            self.writer.stop()
            self.writer.join()
        self.StartAqcuisition.setText("Start Acquisition")

    def toggle_liveHDPC(self, checked):
        if checked:
            self.liveHDPC_thread.start()
            self.StartLiveHDPC.setText("Stop hDPC Live")
        else:
            self.liveHDPC_thread.stop()
            self.StartLiveHDPC.setText("Start hDPC Live")
                       
    def toggle_ref(self, checked):
        def onRefstop():
            self.GetRef.setText("Record reference")
            self.GetRef.setChecked(False)
        if checked:
            self.refacquisition_thread.start()
            self.refacquisition_thread.finished.connect(onRefstop)
            self.GetRef.setText("Stop ref ")
        else:
            self.refacquisition_thread.stop()
            self.GetRef.setText("Record reference")
            
    def toggle_refzStack(self, checked):
        if checked:
            self.ref_zstack.start()
            self.StartAqcuisition.setText("Stop Acquisition")
        else:
            self.ref_zstack.stop()
            self.StartAqcuisition.setText("Start Acquisition")
            
    def toggle_acqAF(self, checked):
        #is_checked = self.uploadRef.isChecked()
        #self.Acq_AF.set_ref_upload(is_checked)
        nIm = self.AcqAF_nIm.value()
        freq = self.n_freqAcqAF.value()
        if checked:
            self.Acq_AF = ACQautofocus(nIm,freq, ref_image = self.ref_image, ref_zStack = self.ref_zStack,  main_window=self)
            self.Acq_AF.logs_signal.connect(self.append_logs)
            self.Acq_AF.frame_received.connect(self.update_frame)
            self.Acq_AF.start()
            self.Acq_AF.data1_ready.connect(self.plot_window.update_plot1)
            self.Acq_AF.data2_ready.connect(self.plot_window.update_plot2)
            self.Acq_AF.started.connect(self.plot_window.show)
            self.Acq_AF.started.connect(self.plot_window.reset)
            self.degree.valueChanged.connect(self.Acq_AF.update_degree_value)
            self.StartAqcuisition.setText("Stop Acquisition")
        else:
            self.Acq_AF.stop()
            self.StartAqcuisition.setText("Start Acquisition")

    def reset_autolevels(self):
        self.reset_autolevels = True
    
    def update_frame(self, frame):
        self.current_frame = frame
        if self.image_view.image is None or self.reset_autolevels:
            self.image_view.setImage(self.current_frame, autoLevels=True, autoRange=False)
            self.reset_autolevels = False
        else:
            self.image_view.setImage(self.current_frame, autoLevels=False, autoRange=False)

    def mouse_moved(self, pos):
        vb = self.image_view.getView()
        mouse_point = vb.mapSceneToView(pos)
        x, y = int(mouse_point.x()), int(mouse_point.y())

        if self.current_frame is not None:
            if 0 <= x < self.current_frame.shape[1] and 0 <= y < self.current_frame.shape[0]:
                val = self.current_frame[y, x]
                self.pos_info_label.setText(f"X: {x}, Y: {y}, Val: {val}")
            else:
                self.pos_info_label.setText("Out of bounds")
                
    
    def set_exposure(self, value):
        mmc.setExposure(value)
    
    def illumination_pattern_change(self, which_pattern):
        if which_pattern == '4 LED':
            self.pattern = (["4","8","16","32"])
            self.textBrowser.append(" 4 LEDs pattern")
        elif which_pattern == '2 LED':
            self.pattern = (["4","8"])
            self.textBrowser.append(" 2 LEDs pattern")
            
    
    def illumination_change(self, led_name):
        if led_name == 'all LEDs':   # it checks which option was choosed in the GUI
            mmc.setProperty("Arduino-Switch",  "State", '60')    # to activae all leds 
            self.textBrowser.append("all leds")
        elif led_name == 'LED 1':
            self.textBrowser.append(" led 1")
            mmc.setProperty("Arduino-Switch",  "State", '4')
        elif led_name == 'LED 2':
            self.textBrowser.append(" led 2")
            mmc.setProperty("Arduino-Switch",  "State", '8')
        elif led_name == 'LED 3':
            self.textBrowser.append(" led 3")
            mmc.setProperty("Arduino-Switch",  "State", '16')
        elif led_name == 'LED 4':
            self.textBrowser.append(" led 4")
            mmc.setProperty("Arduino-Switch",  "State", '32')

    def interrupt_all_threads(self):
        """Stop any running acquisition thread safely."""
        self.append_logs("Interrupt requested — stopping all acquisition threads...")
    
        threads = [
            getattr(self, 'live_thread', None),
            getattr(self, 'liveHDPC_thread', None),
            getattr(self, 'refacquisition_thread', None),
            getattr(self, 'acquisition_thread', None),
            getattr(self, 'ref_zstack', None),
            getattr(self, 'Acq_AF', None),
        ]
    
        for t in threads:
            if t and t.isRunning():
                try:
                    t.stop()  # Set running=False
                    t.wait(2000)  # Wait 2s max to join
                    self.append_logs(f"Stopped {t.__class__.__name__}")
                except Exception as e:
                    self.append_logs(f"Error stopping {t.__class__.__name__}: {e}")
    
        # Reset UI buttons and text
        self.StartLive.setChecked(False)
        self.StartAqcuisition.setChecked(False)
        self.StartLiveHDPC.setChecked(False)
        self.GetRef.setChecked(False)
        self.RefzStack.setChecked(False)
        self.AcqAF.setChecked(False)
    
        self.StartLive.setText("Start Live")
        self.StartAqcuisition.setText("Start Acquisition")
        self.StartLiveHDPC.setText("Start hDPC Live")
        self.GetRef.setText("Record Reference")
        self.RefzStack.setText("Record zStack")
        self.AcqAF.setText("Start Acquisition (AF)")
    
        self.append_logs("All threads stopped.")

    @pyqtSlot(str)
    def append_logs(self, text):
        self.textBrowser.append(text)


    def closeEvent(self, event):
        self.live_thread.stop()
        event.accept()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
