import cv2
import numpy as np
from utils.drawer import Drawer
from utils.xai_tool import XAITool

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
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
model = VGG16(weights = 'imagenet')
input_size = (224, 224)
decode_predictions = decode_predictions
preprocess_input = preprocess_input

# Init drawer and xai_tool
drawer = Drawer()
xai_tool = XAITool(model, input_size, decode_predictions, preprocess_img_func=preprocess_input)
cap = cv2.VideoCapture(0)
while(True):
    ret_run, frame = cap.read()
    preds, heatmap, activations = xai_tool.vidCapRun(frame, -1)
    drawer.draw(frame, heatmap, activations, preds, -1)

# TEMP
# ret_run, frame = cap.read()
# img_tensor = xai_tool.preprocessImg(frame, input_size, preprocess_input)
# preds = xai_tool.getPrediction(model, img_tensor, decode_predictions)
# print(preds)

