import cv2
from darkflow.net.build import TFNet
import matplotlib.pyplot as plt

#This is an ipython command, whatever that means.
#%config InlineBackend.figure_format = 'svg'


'''
##########################################
    Step 1: Create the training model
##########################################
'''
#Some variables for training the model
options = {"model": "cfg/tiny-yolo.cfg", 
           "load": "bin/yolov2-tiny-voc.weights",  #Start training on original model
           #"load": -1,  #Resume loading from most recently trained weights
           "batch": 1,  #Oringially 2
           "epoch": 20000,   #Originally 5
           "train": True,
           "annotation": "../Kangaroo/annots/",
           "dataset": "../Kangaroo/images/",
           "gpu": 1.0
}

#Create a tensorflow network object
tfnetTrain = TFNet(options)

#106.0 is the starting loss, so adjust epoch and stuff accordingly
#Goes down by around 13 every 5 epochs
#Train the model with the options it was created with
tfnetTrain.train()

print("Done training yo.")