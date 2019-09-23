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

  def showLabels(self, predictions):
    # Setting up canvas and defaults for the text
    font = cv2.FONT_HERSHEY_SIMPLEX
    btm_left = (10, 790)
    font_scale = 1
    font_color = (255,255,255)
    line_type = 2
    # Formating text to show
    txt_to_show = 'Prediction: {} Confidence Score: {}'.format(predictions[1], predictions[2])
    cv2.putText(self.mask, txt_to_show,
    btm_left,
    font,
    font_scale,
    font_color,
    line_type
    )
    

  def draw(self, input_img, heatmap, activations, predictions):
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
    # If there is more than 100 activations to show
    else:
      for _row in range(10):
        for _col in range(10):
          act_to_show = resized_activations[_row * 10 + _col]
          self.mask[(act_len * _row):(act_len* (_row + 1)), (400 + act_len * _col):(400 + act_len * (_col + 1)), :3] = act_to_show   
    # Adding the prediction label
    self.showLabels(predictions)
    cv2.imshow('XAI_tool', self.mask)
    cv2.waitKey(100)