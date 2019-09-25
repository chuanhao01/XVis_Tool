import cv2
import numpy as np
from utils.drawer import Drawer
from utils.xai_tool import XAITool

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

from time import time

# mask = np.zeros((400, 400, 3), dtype='uint8')
# # print(mask.shape)

# img2d = np.zeros((100, 100), dtype='uint8')

# img2d[:50, :50] = 255
# img3d = []
# for c_channel in range(3):
#     img3d.append(img2d)

# print(img3d)
# img3d = np.array(img3d)
# img3d = np.transpose(img3d)
# print(img3d.shape)

# mask[50:150, 50:150, :3] = img3d 

# cv2.imshow('hello', mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# x = 80 
# num_of_rows = x // 10
# col_last = x - num_of_rows * 10
# print(col_last)

# for _row in range(num_of_rows):
#     for _col in range(10):
#         # DO something
#         pass
    
# for _col in range(col_last):
#     # Do something
#     pass

# # Init drawer class
# drawer = Drawer()
# # Init XAIToolHeatmap class
# img_path = 'Sample_Images/cat_2.jpg'
# model = VGG16(weights = 'imagenet')
# input_size = (224, 224)
# xai_tool = XAITool(img_path, model, input_size)
# activations = xai_tool.xai_activations.runTool(0)

# canvas = np.zeros((800, 1200, 3), dtype='uint8')
# act_len = 150
# for row in range(5):
#     for col in range(2):
#         act_index = row * 5 + col
#         activation = activations[act_index]
#         activation = cv2.resize(activation, (act_len, act_len))
#         canvas[(act_len * row):(act_len * (row + 1)), (act_len * col):(act_len * (col + 1)), :3] = activation

# cv2.imshow('Debugging', canvas)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

print(time())
