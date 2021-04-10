import cv2
from darkflow.net.build import TFNet
import matplotlib.pyplot as plt
import numpy as np


def pred_img(options):
    #This loads pre-trained parameters from the checkpoint that we just specified in options.
    tfnetPredict = TFNet(options)
    tfnetPredict.load_from_ckpt()

    #We will take one of the images that has not been used for training to predict the class, bounding box and confidence
    original_img = cv2.imread("testimgs/soccer2.jpg")

    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    result = tfnetPredict.return_predict(original_img)
    print(result)

    # pull out some info from the results
    for i in range(0, len(result)):
        tl = (result[i]['topleft']['x'], result[i]['topleft']['y'])
        br = (result[i]['bottomright']['x'], result[i]['bottomright']['y'])
        label = result[i]['label']  # add the box and label and display it
        img = cv2.rectangle(original_img, tl, br, (0, 255, 0), 7)
        img = cv2.putText(
            img, label, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
    plt.imshow(original_img)
    plt.show()



'''
####################################################################
    Step 2: Create the predicting model (Step 1 is found in train.py)
####################################################################
'''

#Create another tensorflow network object to predict what is in the image
options = {
    'model': 'cfg/model_608_coco/yolov2-1c.cfg',
    
    'backup': 'ckpt/',
    'load': 2600,           #Which training checkpoint should this model be loaded from (needs something like tiny-yolo-10.meta to work)
    
    'threshold': 0.0011,   #0.3 default confidence
    # 'gpu': 0.45
}

#Time to predict whats in an image
pred_img(options)