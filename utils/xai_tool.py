import cv2

from .xai_heatmap import XAIHeatmap
from .xai_activations import XAIActivations

import numpy as np
import re

from keras.preprocessing import image

class XAITool:
    def __init__(self, model, input_size, decoder_func, preprocess_img_func = None):
        # Initialising vars
        self.model = model
        self.input_size = input_size
        self.decoder_func = decoder_func
        # Special case for the optional preprocessing function
        if (preprocess_img_func):
            self.preprocess_img_func = preprocess_img_func
        # Getting the conv layers of the model
        self.layers = self.getLayers(self.model)
        # Setting up the instances of the heatmap and activation tool
        self.xai_heatmap = XAIHeatmap(self.model, self.layers)
        self.xai_activations = XAIActivations(self.model, self.layers)
            
    # For still img takes in the path and loads it into the class, then returns the cv2 imread of the image
    def setStillImg(self, img_path):
        # Loading img in correct img space
        self.still_img = image.load_img(img_path, target_size=self.input_size)
        # Turning the image into a np array
        self.still_img = image.img_to_array(self.still_img)
        # Getting the cv2 img to show
        self.still_cv2_img = cv2.imread(img_path)
        return self.still_cv2_img

    # General function to take in an input of a np image, then resize and preprocess
    # Note: this has to be a RGB channel image, take note esp for cv2
    # img = np.array, input_size = tuple
    def preprocessImg(self, img, input_size, preprocess_img_func = None):
        # Resizing the img to input_size if not already
        img_tensor = cv2.resize(img, input_size)
        # Making the img into a 4-D batch for keras
        img_tensor = np.expand_dims(img_tensor, axis=0)
        # Special code in the case a preprocessing function is passed in
        if(preprocess_img_func):
            img_tensor = preprocess_img_func(img_tensor)
        return img_tensor

    # Method to get a list of the predictions
    def getPrediction(self, model, img_tensor, decoder_func):
        preds = model.predict(img_tensor)
        preds = decoder_func(preds, top=1)[0][0]
        return preds

    # Gets and sets the layers of the model in the class, also returns the layer when called
    def getLayers(self, model):
        # Getting the layers from the model
        layers = model.layers

        # Setting up the array that contains the configs
        layer_configs = []

        for layer in layers:
            layer_configs.append(layer.get_config())

        layer_configs = np.array(layer_configs)

        # Setting up the regex and filter
        pattern = re.compile('conv.*')
        layer_names = []

        for layer in layer_configs:
            if pattern.search(layer['name']):
                layer_names.append(layer['name'])

        return layer_names

    def vidCapRun(self, frame, selected_layer):
        # Renaming the var
        cv2img = frame
        # Converting the cv2 input from the webcam into RGB channels
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Preprocessing the image
        if(self.preprocess_img_func):
            img_tensor = self.preprocessImg(img, self.input_size, preprocess_img_func=self.preprocess_img_func)
        else:
            img_tensor = self.preprocessImg(img, self.input_size)
        # Getting the predictions based on the image and model
        preds = self.getPrediction(self.model, img_tensor, self.decoder_func)
        # Getting the heatmap and activations
        heatmap = self.xai_heatmap.runTool(cv2img, img_tensor, selected_layer)
        activations = self.xai_activations.runTool(img_tensor, selected_layer)
        return preds, heatmap, activations
