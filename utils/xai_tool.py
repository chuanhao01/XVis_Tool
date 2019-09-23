import cv2

from .xai_heatmap import XAIHeatmap
from .xai_activation import XAIActivations

import numpy as np

from keras.preprocessing import image

class XAITool:
    def __init__(self, img_path, model, input_size, preprocess_img_func = None):
        # Initialising vars
        self.img_path = img_path
        self.model = model
        self.input_size = input_size
        if (preprocess_img_func):
            self.preprocess_img_func = preprocess_img_func
            self.img_tensor = self.preprocessImgs(self.img_path, self.input_size, self.preprocess_img_func)
        else:
            self.img_tensor = self.preprocessImgs(self.img_path, self.input_size)
        self.cv2img = cv2.imread(img_path)
        # Creating an instance of the heatmap class
        self.xai_heatmap = XAIHeatmap(self.cv2img, self.img_tensor, self.model, self.input_size)
        self.xai_activations = XAIActivations(self.img_tensor, self.model, self.xai_heatmap.layers)


    def preprocessImgs(self, img_path, input_size, preprocess_img_func = None):
        # loading the img in from path
        img = image.load_img(img_path, target_size = input_size)
        # changing the image into a keras form bath in 4-D
        img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        if(preprocess_img_func):
            img_tensor = preprocess_img_func(img_tensor)
        return img_tensor
