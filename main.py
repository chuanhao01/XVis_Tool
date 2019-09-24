import cv2
import numpy as np
from utils.drawer import Drawer
from utils.xai_tool import XAITool

from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from time import time



def userSelection(layers):
    while(True):
        selected_layer = input('>>')
        try:
            selected_layer = int(selected_layer)
            # For positive selection of layer
            if(0 < selected_layer < len(layers) + 1):
                return selected_layer - 1
            # For negative indexes
            elif(-(len(layers) + 1) < selected_layer < 0):
                return selected_layer
            else:
                print('Input out of range')
        except ValueError:
            print("Please give a valid input")


# Setting up the input for the tool
# model is a keras model
# input_size is a tuple of (x, y) which contains the dims of the input image of the CNN
# decode_predictions is the function keras uses to decode the prediction
# preprocess_input is the function keras uses to process the images before passing them into the model. NOTE: this is an optional arg
from keras.applications.xception import Xception, decode_predictions, preprocess_input

model = Xception(weights = 'imagenet')
input_size = (299, 299) 
def decode(model, img_tensor):
    preds = model.predict(img_tensor)
    preds = decode_predictions(preds, top=1)[0][0]
    preds = [preds[1], preds[2]]
    return preds
preprocess_input = preprocess_input

# Init drawer and xai_tool
drawer = Drawer()
xai_tool = XAITool(model, input_size, decode, preprocess_input)
cap = cv2.VideoCapture(0)
while(True):
    ret_run, frame = cap.read()
    xai_dict = xai_tool.vidCapRun(frame, -1)
    if('predictions' in xai_dict):
        drawer.singleThread(frame, xai_dict['heatmap'], xai_dict['activations'], xai_tool.layers[-1], -1, xai_dict['predictions'])
    else:
        drawer.singleThread(frame, xai_dict['heatmap'], xai_dict['activations'], xai_tool.layers[-1], -1)
# TEMP
# cap = cv2.VideoCapture(0)
# ret_run, frame = cap.read()
# cv2.imshow('hello', frame)
# cv2.waitKey(100)
# # testing custom models
# from keras.models import load_model
# import keras
# cus_model = load_model('models/TIL_Pose_detector.h5') 
# input_size = (224, 224)

# xai_tool = XAITool(cus_model, input_size, decode_predictions)
# img_tensor = xai_tool.preprocessImg(frame, input_size, preprocess_input)

# pred = cus_model.predict(img_tensor)
# target_names = ['ChairPose', 'ChestBump', 'ChildPose', 'Dabbing', 'EaglePose', 'HandGun', 'HandShake', 'HighKneel', 'HulkSmash', 'KoreanHeart', 'KungfuCrane', 'KungfuSalute', 'Salute', 'Spiderman', 'WarriorPose']
# pred_index = pred.argmax(axis=-1)

# print('Prediction is: {}'.format(target_names[pred_index[0]]))
# print('Score is: {}'.format(pred[0][pred_index[0]]))

# input()

