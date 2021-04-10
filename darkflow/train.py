import cv2
from darkflow.net.build import TFNet
import matplotlib.pyplot as plt



'''
##########################################
    Step 1: Create the training model
##########################################
'''
#Some variables for training the model
options = { "model": "cfg/model_608_coco/yolov2-1c.cfg",     #Training the 1 class model     
            # "load": "bin/yolov2-coco-608x608.weights",   #Start training on original weights
            "load": -1,     #Resume loading from most recently trained weights
            "batch": 1,      #Oringially 2
            "epoch": 20,     #Originally 5
            "train": True,

            # "annotation": "../Soccerball/annots/",
            # "dataset": "../Soccerball/images/",
             "annotation": "../Kangaroo/annots/",
             "dataset": "../Kangaroo/images/",
            "gpu": 1.0,
            "save": 175       # save/batch is how many steps the model saves. 

}

#Create a tensorflow network object
tfnetTrain = TFNet(options)

#Train the model with the options it was created with
tfnetTrain.train()

print("Done training yo.")