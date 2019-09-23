import cv2
import numpy as np
from utils.drawer import Drawer
from utils.xai_tool import XAITool

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

def userSelection():
    while(True):
        selected_layer = input('>>')
        if(selected_layer.isdigit()):
            selected_layer = int(selected_layer)
            if(0 < selected_layer < len(layer_names) + 1):
                return selected_layer - 1
            else:
                print('Selection out of range')
        else:
            print('Plefrom keras.preprocessing import imagese give a valid input')



# Init drawer class
drawer = Drawer()
# Init XAIToolHeatmap class
img_path = 'Sample_Images/rando_person.jpg'
model = VGG16(weights = 'imagenet')
input_size = (224, 224)
xai_tool = XAITool(img_path, model, input_size, preprocess_input)
# For first instance
img = xai_tool.cv2img
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
heatmap = xai_tool.xai_heatmap.runTool(-1)
activations = xai_tool.xai_activations.runTool(-1)
drawer.draw(img, heatmap, activations)
# Setting up the var
layer_names = xai_tool.xai_heatmap.layers
while(True):
    # Printing out the layer names
    print('These are the layers you can select: ')
    for i, layer_name in enumerate(layer_names):
        print(layer_name, i)
        print('Layer ({}): {}'.format(int(i) + 1, layer_name))
        print("Please Select a layer")
    # Getting layer selected
    layer_selection = userSelection()
    new_heatmap = xai_tool.xai_heatmap.runTool(layer_selection)
    new_activations = xai_tool.xai_activations.runTool(layer_selection)
    drawer.draw(img, new_heatmap, new_activations)

