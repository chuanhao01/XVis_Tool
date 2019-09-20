import cv2
import numpy as np

class Drawer:
  def __init__(self):
    self.mask = None

  # To resize the input and heatmap imgs
  # Params is a numpy array
  def resizeDisplayImg(self, img_to_resize):
    # The len of sq where the img will be in
    sq_len = 400
    dims = img_to_resize.shape[:2]
    if(dims[0] > dims[1]):
      new_dims = (400, int(dims[1] * 400/dims[0]))
    else:
      new_dims = (int(dims[0] * 400/dims[1]), 400)
    resized_img = cv2.resize(img_to_resize, new_dims)
    return resized_img

  def draw(self, input_img, heatmap, activations):
    # Creating the black BG
    self.mask = np.zeros((800, 800, 3), dtype='uint8')
    resized_input = self.resizeDisplayImg(input_img)
    resized_heatmap = self.resizeDisplayImg(heatmap)
    # Adding input img
    self.mask[:resized_input.shape[0], :resized_input.shape[1], :resized_input.shape[2]] = resized_input
    # Adding heatmap
    self.mask[400:400 + resized_heatmap.shape[0], :resized_heatmap.shape[1], :resized_heatmap.shape[2]] = resized_heatmap
    cv2.imshow('XAI_tool', self.mask)
    cv2.waitKey(100)