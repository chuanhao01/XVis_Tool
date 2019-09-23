import cv2
import numpy as np

class Drawer:
  def __init__(self):
    self.mask = None

  # To resize the input and heatmap imgs
  # Params is a numpy array
  def resizeDisplayImg(self, img_to_resize, sq_len):
    # The len of sq where the img will be in
    dims = img_to_resize.shape[:2]
    if(dims[0] > dims[1]):
      new_dims = (400, int(dims[1] * 400/dims[0]))
    else:
      new_dims = (int(dims[0] * 400/dims[1]), 400)
    resized_img = cv2.resize(img_to_resize, new_dims)
    return resized_img

  def draw(self, input_img, heatmap, activations):
    # Creating the black BG
    self.mask = np.zeros((800, 1200, 3), dtype='uint8')
    resized_input = self.resizeDisplayImg(input_img, 400)
    resized_heatmap = self.resizeDisplayImg(heatmap, 400)
    # Adding input img
    self.mask[:resized_input.shape[0], :resized_input.shape[1], :resized_input.shape[2]] = resized_input
    # Adding heatmap
    self.mask[400:400 + resized_heatmap.shape[0], :resized_heatmap.shape[1], :resized_heatmap.shape[2]] = resized_heatmap
    # For adding activations
    # To get the resized activations
    resized_activations = []
    act_len = 80 # Length of each activation in px
    for activation in activations:
      resized_activation = cv2.resize(activation, (act_len, act_len)) 
      resized_activations.append(resized_activation)
    # To place the activations 
    if(len(resized_activations) < 100):
      num_of_acts = len(resized_activations)
      row_num = num_of_acts // 10
      col_last = num_of_acts - row_num * 10
      # For the full shown rows
      for _row in range(row_num):
        for _col in range(10):
          act_to_show = resized_activations[_row * 10 + _col]
          self.mask[(act_len * _row):(act_len* (_row + 1)), (400 + act_len * _col):(400 + act_len * (_col + 1)), :3] = act_to_show   
      # For the last row
      if(col_last > 0):
        for _col in range(col_last):
          act_to_show = resized_activations[row_num * 10 + _col]
          self.mask[(act_len * row_num):(act_len* (row_num + 1)), (400 + act_len * _col):(400 + act_len * (_col + 1)), :3] = act_to_show   
    else:
      for _row in range(10):
        for _col in range(10):
          act_to_show = resized_activations[_row * 10 + _col]
          self.mask[(act_len * _row):(act_len* (_row + 1)), (400 + act_len * _col):(400 + act_len * (_col + 1)), :3] = act_to_show   
    # Old
    # for row_num in range(1):
    #   for col_num in range(1):
    #     act_to_show = resized_activations[row_num * 10 + col_num]
    #     self.mask[(40 * row_num):(40 * (row_num + 1)), (400 + 40 * col_num):(400 + 40 * (col_num + 1)), :3] = act_to_show   
    cv2.imshow('XAI_tool', self.mask)
    cv2.waitKey(100)