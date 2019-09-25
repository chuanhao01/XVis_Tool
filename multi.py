import cv2
import numpy as np
from utils.drawer import Drawer
from utils.xai_tool import XAITool

from time import time

# For multi-threading
from threading import Thread, Lock

# Setting up the input for the tool
# model is a keras model
# input_size is a tuple of (x, y) which contains the dims of the input image of the CNN
# decode_predictions is the function keras uses to decode the prediction
# preprocess_input is the function keras uses to process the images before passing them into the model. NOTE: this is an optional arg

# Initialising relevant classes and shared vars
drawer = Drawer()
cap = cv2.VideoCapture(0)
shared_dict = {
    'frame' : None,
    'heatmap': None,
    'activations': None,
    'predictions': None, 
    'select_layers_list': None,
    'selected_layer': -1,
}
data_lock = Lock()

# Util function to validate the user selection of the layer
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


# Thread function to change the currently selected layer 
# The thread is spinned up in the xai thread
def changeSelectedLayer(layers):
    # Function in another thread to allow the user to select the thread that he wants running
    while(True):
        print('These are the layers you can select: ')
        for i, layer in enumerate(layers):
            print('Layer ({}): {}'.format(i + 1, layer)) 
        selected_layer = getSelectedLayer(layers)
        with data_lock:
            shared_dict['selected_layer'] = selected_layer

# The thread function for doing xai
def xaiProcessing():
    # Need to import the keras mdoel in the thread itself
    from keras.applications.xception import Xception, decode_predictions, preprocess_input
    from keras.models import load_model
    # model = Xception(weights = 'imagenet')
    model = load_model('models/best_vgg_based_mnist.h5')
    input_size = (48, 48) 
    target_labels = [str(i) for i in range(10)]
    preprocess_input = None 

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

    xai_tool = XAITool(model, input_size, decoder_func = decode, preprocess_img_func=preprocess_input)

    # Spawning a thread to change selected layers
    change_layer_thread = Thread(target = changeSelectedLayer, args=([xai_tool.layers]))
    change_layer_thread.start()

    while(True):    
        if(shared_dict['frame'] is not None):
            drawer.resetMask()
            with data_lock:
                grabbed_frame = shared_dict['frame']
                # grabbed_frame = cv2.cvtColor(grabbed_frame, cv2.COLOR_BGR2GRAY)
                selected_layer_num = shared_dict['selected_layer']
            xai_dict = xai_tool.vidCapRun(grabbed_frame, selected_layer_num)
            select_layers_list = [xai_tool.layers[selected_layer_num], selected_layer_num]
            for key in xai_dict:
                shared_dict[key] = xai_dict[key]
            shared_dict['select_layers_list'] = select_layers_list


xai_process_thread = Thread(target=xaiProcessing)
xai_process_thread.start()

while(True):
    _, frame = cap.read()
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # gray = np.stack([gray, gray, gray], axis=2)
    shared_dict['frame'] = frame 
    drawer.drawOriPic(frame) 
    if(shared_dict['heatmap'] is not None and shared_dict['activations'] is not None and shared_dict['select_layers_list'] is not None):
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
 


