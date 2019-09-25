import cv2
import numpy as np
from utils.drawer import Drawer
from utils.xai_tool import XAITool

from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from time import time

# For multi-threading
from threading import Thread, Lock

# Setting up the input for the tool
# model is a keras model
# input_size is a tuple of (x, y) which contains the dims of the input image of the CNN
# decode_predictions is the function keras uses to decode the prediction
# preprocess_input is the function keras uses to process the images before passing them into the model. NOTE: this is an optional arg
from keras.applications.vgg16 import VGG16, decode_predictions, preprocess_input

from keras.applications.xception import Xception, decode_predictions, preprocess_input

model = Xception(weights = 'imagenet')
model._make_predict_function()
model.predict(np.zeros((1, 299, 299, 3)))
input_size = (299, 299) 
def decode(model, img_tensor):
    preds = model.predict(img_tensor)
    preds = decode_predictions(preds, top=1)[0][0]
    preds = [preds[1], preds[2]]
    return preds
preprocess_input = preprocess_input


# Declaring input vars
# model = VGG16(weights='imagenet')
# input_size = (224, 224)
# def decode(model, img_tensor):
#     preds = model.predict(img_tensor)
#     preds = decode_predictions(preds, top=1)[0][0]
#     preds = [preds[1], preds[2]]
#     return preds
# preprocess_input = preprocess_input

# Initialising relevant classes and shared vars
xai_tool = XAITool(model, input_size, decode, preprocess_img_func=preprocess_input)
drawer = Drawer()
cap = cv2.VideoCapture(0)
shared_dict = {
    'frame' : None,
    'heatmap': None,
    'activations': None,
    'predictions': None, 
    'select_layers_list': None,
}
data_lock = Lock()

def xaiProcessing():
    while(True):    
        if(shared_dict['frame'] is not None):
            with data_lock:
                grabbed_frame = shared_dict['frame']
            selected_layer_num = 10
            xai_dict = xai_tool.vidCapRun(grabbed_frame, selected_layer_num)
            select_layers_list = [xai_tool.layers[selected_layer_num], selected_layer_num]
            for key in xai_dict:
                shared_dict[key] = xai_dict[key]
            shared_dict['select_layers_list'] = select_layers_list

xai_process_thread = Thread(target=xaiProcessing)
xai_process_thread.start()

while(True):
    _, frame = cap.read()
    shared_dict['frame'] = frame
    drawer.drawOriPic(frame) 
    if(shared_dict['heatmap'] and shared_dict['activations'] and shared_dict['select_layersList']):
        heatmap = shared_dict['heatmap']
        activations = shared_dict['activations']
        select_layers_list = shared_dict['select_layers_list']
        preds = shared_dict['predictions']
        drawer.multiTemp(heatmap, activations, select_layers_list, preds = preds)
        with data_lock:
            shared_dict['heatmap'] = None
            shared_dict['activations'] = None
            shared_dict['predictions'] = None
            shared_dict['select_layers_list'] = None
    cv2.imshow('XAI_multi', drawer.mask)
    cv2.waitKey(100)
    # drawer.resetMask()


