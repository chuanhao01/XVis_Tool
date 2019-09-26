import cv2
import numpy as np
from utils.drawer import Drawer
from utils.xai_tool import XAITool

def getSelectedLayer(layers):
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
from keras.models import load_model

# Changes come below here 

model = Xception(weights = 'imagenet')
img_path = 'Sample_Images/doberman_1.jpg'
# model = load_model('models/best_xception_based_mnist.h5')
input_size = (299, 299) 
target_labels = None
# target_labels = [str(i) for i in range(10)]
preprocess_input = preprocess_input 

# Wrapper so as to automatically create the decoder function
def createDecoder(target_labels):
    if(target_labels is not None):
        def decode(model, img_tensor):
            preds = model.predict(img_tensor)
            # target_labels = target_labels
            pred_index = preds.argmax(axis=-1)[0]
            return [target_labels[pred_index], preds[0][pred_index]]    
    else:
        def decode(model, img_tensor):
            preds = model.predict(img_tensor)
            preds = decode_predictions(preds, top=1)[0][0]
            preds = [preds[1], preds[2]]
            return preds
    return decode
decode = createDecoder(target_labels)

# Init classes
drawer = Drawer()
cap = cv2.VideoCapture(0)
xai_tool = XAITool(model, input_size, decoder_func = decode, preprocess_img_func=preprocess_input)

# Showing the default 
ori_img = xai_tool.setStillImg(img_path) 
xai_dict = xai_tool.stillImgRun(-1)
layers = xai_tool.layers

drawer.singleThread(ori_img, xai_dict['heatmap'], xai_dict['activations'], [layers[-1], -1], xai_dict['predictions'])
cv2.imshow('XAI_Single', drawer.mask)
cv2.waitKey(100)
while(True):
    print('These are the layers you can select: ')
    for i, layer in enumerate(layers):
        print('Layer ({}): {}'.format(i + 1, layer)) 
    selected_layer = getSelectedLayer(layers)
    xai_dict = xai_tool.stillImgRun(selected_layer)
    drawer.singleThread(ori_img, xai_dict['heatmap'], xai_dict['activations'], [layers[selected_layer], selected_layer], xai_dict['predictions'])
    cv2.imshow('XAI_Single', drawer.mask)
    cv2.waitKey(100)