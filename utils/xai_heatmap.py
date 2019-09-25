import pandas as pd
import numpy as np
import seaborn as sns

from keras.applications.vgg16 import VGG16
import matplotlib.image as mpimg
from keras import backend as K
import matplotlib.pyplot as plt

from keras.preprocessing import image
import cv2
import re

from keras.applications.vgg16 import VGG16

class XAIHeatmap:
  def __init__(self, model, layers, graph = None):
    self.model = model
    self.layers = layers
    self.graph = None
    if(graph):
      self.graph = graph
  
  # High level wrapping function to run and get a heatmap
  def runTool(self, cv2img, img_tensor, layer_num):
    heatmap = self.camXAITool(cv2img, img_tensor, self.model, self.layers[layer_num])
    return heatmap

  def camXAITool(self, cv2img, img_tensor, model, layer_name):
    # Getting the predictions from the model
    # preds is the coded preds 

    # Debugging time code
    import time
    tic = time.time()

    if(self.graph):
      with self.graph.as_default():
        preds = model(img_tensor)
    else:
      preds = model.predict(img_tensor)

    tic3 = time.time()
    
    # Getting the tensor of all weights of the final dense layer
    output = model.output
    # getting the indice/position of the weight that contributed the most to the classification
    argmax = np.argmax(preds[0])
    # isolating the weight that contributed the most to the classification in the output tensor
    output = output[:, argmax]
    # getting the conv_layer to apply the heatmap to
    conv_layer = model.get_layer(layer_name).output
    # Getting the gradients of the most activated weight w.r.t. the conv_layer
    # Also getting the mean of the gradients per feature map 
    grads = K.gradients(output, conv_layer)[0]
    pooled_grads = K.mean(grads, axis = (0, 1, 2))
    # Making the Keras function
    iterate = K.function([model.input], [pooled_grads, conv_layer])
    # Getting the pooled_grads and conv layer w.r.t. the input img
    pooled_grads_value, conv_layer_output_layer = iterate([img_tensor])
    for i in range(len(pooled_grads_value)):
      conv_layer_output_layer[0, :, :, i] *= pooled_grads_value[i]
    # Generating the heatmap based on the grads and conv_layer
    heatmap = np.mean(conv_layer_output_layer, axis=-1)[0]
    # Im still confused about the np.maximum funct
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    tic4 = time.time()

    # Loading the img using cv2
    img = cv2img 
    # resizing and normalizing the heatmap
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(heatmap * 255)
    # applying the heatmap
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    hif = 0.8
    superimposed_image = heatmap * hif + img
    superimposed_image /= np.max(superimposed_image)
    superimposed_image *= 255.

    tic5  = time.time()
    
    print('forward pass: {}s'.format(tic3 - tic))
    print('camming: {}s'.format(tic4 - tic3))
    print('drawing: {}s'.format(tic5 - tic4))

    return superimposed_image  
  
