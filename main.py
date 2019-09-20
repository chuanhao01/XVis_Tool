import cv2
import numpy as np
from utils.drawer import Drawer
from utils.xai_heatmap_tool import XAIToolHeatmap

from keras.applications.vgg16 import VGG16

def userSelection():
    while(True):
        selected_layer = input('>>')
        if(selected_layer.isdigit()):
            selected_layer = int(selected_layer)
            if(0 < selected_layer < len(layer_names)):
                return selected_layer - 1
            else:
                print('Selection out of range')
        else:
            print('Please give a valid input')


# Init drawer class
drawer = Drawer()
# Init XAIToolHeatmap class
img_path = 'Sample_Images/cat_1.jpg'
model = VGG16(weights = 'imagenet')
input_size = (224, 224)
xai_heatmap_tool = XAIToolHeatmap(img_path, model, input_size)
# For first instance
img = cv2.imread(img_path)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
heatmap = xai_heatmap_tool.runTool(-1)
drawer.draw(img, heatmap, 0)
# Setting up the var
layer_names = xai_heatmap_tool.layers
while(True):
    # Printing out the layer names
    print('These are the layers you can select: ')
    for i, layer_name in enumerate(layer_names):
        print(layer_name, i)
        print('Layer ({}): {}'.format(int(i) + 1, layer_name))
        print("Please Select a layer")
    # Getting layer selected
    layer_selection = userSelection()
    new_heatmap = xai_heatmap_tool.runTool(layer_selection)
    drawer.draw(img, new_heatmap, 0)

