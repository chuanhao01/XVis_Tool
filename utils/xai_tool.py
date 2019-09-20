import cv2

from .xai_heatmap import XAIHeatmap

import numpy as np

from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

class XAITool():
    def __init__(self, img_path, model, input_size):
        # Initialising vars
        self.img_path = img_path
        self.model = model
        self.input_size = input_size
        self.img_tensor = self.preprocessImgs(self.img_path, self.input_size)
        self.cv2img = cv2.imread(img_path)
        # Creating an instance of the heatmap class
        self.xai_heatmap = XAIHeatmap(self.cv2img, self.img_tensor, self.model, self.input_size)

    def preprocessImgs(self, img_path, input_size):
        # loading the img in from path
        img = image.load_img(img_path, target_size = input_size)
        # changing the image into a keras form bath in 4-D
        img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor = preprocess_input(img_tensor)
        return img_tensor
