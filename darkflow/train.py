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
options = { "model": "cfg/model_608_coco/yolov2-1c.cfg",     #Training the 1 class model     
            "load": "bin/yolov2-coco-608x608.weights",   #Start training on original weights
            # "load": -1,     #Resume loading from most recently trained weights
            "batch": 8,      #Oringially 2
            "epoch": 100,     #Originally 5
            "train": True,

            "annotation": "../Soccerball/annots/",
            "dataset": "../Soccerball/images/",
            #  "annotation": "../Kangaroo/annots/",
            #  "dataset": "../Kangaroo/images/",
            "gpu": 1.0,
            "save": 256       # save/batch is how many steps the model saves. 

}

#Create a tensorflow network object
tfnetTrain = TFNet(options)

#106.0 is the starting loss, so adjust epoch and stuff accordingly
#Goes down by around 13 every 5 epochs
#Train the model with the options it was created with
tfnetTrain.train()

print("Done training yo.")