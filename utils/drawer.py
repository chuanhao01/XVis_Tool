import cv2
import numpy as np

class Drawer:
  def __init__(self):
    self.mask = np.zeros((800, 1200, 3), dtype='uint8')

  # To resize the input and heatmap imgs
  # Params is a numpy array and the len of the square you want to resize to in px
  def resizeDisplayImg(self, img_to_resize, sq_len):
    # The len of sq where the img will be in
    dims = img_to_resize.shape[:2]
    if(dims[0] > dims[1]):
      new_dims = (sq_len, int(dims[1] * sq_len/dims[0]))
    else:
      new_dims = (int(dims[0] * sq_len/dims[1]), sq_len)
    resized_img = cv2.resize(img_to_resize, new_dims)
    return resized_img

  # Cords in the form of a 2-D tuple of the top-left cords
  # cv2img is a np array
  def drawPicOver(self, cv2img, cords):
    top_left_x = cords[1]
    top_left_y = cords[0]
    img_len = cv2img.shape[1]
    img_height = cv2img.shape[0]
    self.mask[top_left_y:(top_left_y + img_height), top_left_x:(top_left_x + img_len), :3] = cv2img

  # Wrappers for the resp drawing
  def drawOriPic(self, ori_img):
    cords = (0, 0)
    ori_img = self.resizeDisplayImg(ori_img, 400)
    self.drawPicOver(ori_img, cords)

  def drawHeatmap(self, heatmap):
    cords = (400, 0)
    heatmap = self.resizeDisplayImg(heatmap, 400)
    self.drawPicOver(heatmap, cords)

  # Special wrapper as the activations come in a list of np arrays
  def drawActivations(self, activations):
    # To resized the activations
    act_len = 80
    resized_acts = [] 
    for activation in activations:
      resized_act = self.resizeDisplayImg(activation, act_len)
      resized_acts.append(resized_act)
    # The offset of the activations
    x_offset = 400
    # If there are lesser than 100 activations to show
    if(len(resized_acts) < 100):
      num_of_acts = len(resized_acts)
      row_num = num_of_acts // 10
      col_last = num_of_acts - row_num * 10
      # For fully shown rows
      for _row in range(row_num):
        for _col in range(10):
          act_to_show = resized_acts[_row * 10 + _col]
          cords = (act_len * _row, x_offset + act_len * _col) 
          self.drawPicOver(act_to_show, cords)
      # For last layer
      if(col_last > 0):
        for _col in range(col_last):
          act_to_show = resized_acts[row_num * 10 + _col]
          cords = (act_len * row_num, x_offset + act_len * _col)
          self.drawPicOver(act_to_show, cords)
    # If there are more than 100 activations to show
    else:
      for _row in range(10):
        for _col in range(10):
          act_to_show = resized_acts[_row * 10 + _col]
          cords = (act_len * _row, x_offset + act_len * _col)
          self.drawPicOver(act_to_show, cords)
    
  def drawLayerNum(self, font, btm_left, font_scale, font_color, line_type, draw_layers_list):
    txt_to_show = 'Layer: {} Number: {}'.format(draw_layers_list[0], draw_layers_list[1])
    cv2.putText(self.mask, txt_to_show,
    btm_left,
    font,
    font_scale,
    font_color,
    line_type
    )
  
  # Important note, the preds must be in the form, [predictions, confidence score]
  def drawPredictions(self, font, btm_left, font_scale, font_color, line_type, preds):
    txt_to_show = 'Prediction: {} Score: {}'.format(preds[0], preds[1])
    cv2.putText(self.mask, txt_to_show,
    btm_left,
    font,
    font_scale,
    font_color,
    line_type
    )

  def multiTemp(self, heatmap, activations,  select_layers_list, preds = None):
    self.drawHeatmap(heatmap)
    self.drawActivations(activations)
    # Setting up for the text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)
    line_type = 2
    cord_layer = (10, 750)
    self.drawLayerNum(font, cord_layer, font_scale, font_color, line_type, select_layers_list)
    if(preds):
      cord_pred = (10, 790)
      self.drawPredictions(font, cord_pred, font_scale, font_color, line_type, preds)
  
  def resetMask(self):
    self.mask = np.zeros((800, 1200, 3), dtype='uint8')

  def singleThread(self, input_img, heatmap, activations, selected_layer, layer_num, preds = None):
    self.drawOriPic(input_img)
    self.drawHeatmap(heatmap)
    self.drawActivations(activations)
    # Setting up for the text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)
    line_type = 2
    cord_layer = (10, 750)
    self.drawLayerNum(font, cord_layer, font_scale, font_color, line_type, (selected_layer, layer_num))
    if(preds):
      cord_pred = (10, 790) 
      self.drawPredictions(font, cord_pred, font_scale, font_color, line_type, preds)
    cv2.imshow('XAI Tool', self.mask)
    cv2.waitKey(100)
    self.mask = np.zeros((800, 1200, 3), dtype='uint8')

      
    

# class Drawer:
#   def __init__(self):
#     self.mask = None

#   # To resize the input and heatmap imgs
#   # Params is a numpy array and the len of the square you want to resize to in px
#   def resizeDisplayImg(self, img_to_resize, sq_len):
#     # The len of sq where the img will be in
#     dims = img_to_resize.shape[:2]
#     if(dims[0] > dims[1]):
#       new_dims = (400, int(dims[1] * 400/dims[0]))
#     else:
#       new_dims = (int(dims[0] * 400/dims[1]), 400)
#     resized_img = cv2.resize(img_to_resize, new_dims)
#     return resized_img

#   def showImgText(self, predictions, selected_layer):
#     # Setting up defaults for the text
#     # For the preds
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     btm_left = (10, 790)
#     font_scale = 1
#     font_color = (255,255,255)
#     line_type = 2
#     # Formating text to show
#     txt_to_show = 'Prediction: {} Confidence Score: {}'.format(predictions[1], predictions[2])
#     cv2.putText(self.mask, txt_to_show,
#     btm_left,
#     font,
#     font_scale,
#     font_color,
#     line_type
#     )
#     # For the selected layer
#     btm_left = (10, 750)
#     txt_to_show = 'Layer {}'.format(selected_layer)
#     cv2.putText(self.mask, txt_to_show,
#     btm_left,
#     font,
#     font_scale,
#     font_color,
#     line_type
#     )
    

#   def draw(self, input_img, heatmap, activations, predictions, selected_layer):
#     # Creating the black BG
#     self.mask = np.zeros((800, 1200, 3), dtype='uint8')
#     resized_input = self.resizeDisplayImg(input_img, 400)
#     resized_heatmap = self.resizeDisplayImg(heatmap, 400)
#     # Adding input img
#     self.mask[:resized_input.shape[0], :resized_input.shape[1], :resized_input.shape[2]] = resized_input
#     # Adding heatmap
#     self.mask[400:400 + resized_heatmap.shape[0], :resized_heatmap.shape[1], :resized_heatmap.shape[2]] = resized_heatmap
#     # For adding activations
#     # To get the resized activations
#     resized_activations = []
#     act_len = 80 # Length of each activation in px
#     for activation in activations:
#       resized_activation = cv2.resize(activation, (act_len, act_len)) 
#       resized_activations.append(resized_activation)
#     # To place the activations 
#     if(len(resized_activations) < 100):
#       num_of_acts = len(resized_activations)
#       row_num = num_of_acts // 10
#       col_last = num_of_acts - row_num * 10
#       # For the full shown rows
#       for _row in range(row_num):
#         for _col in range(10):
#           act_to_show = resized_activations[_row * 10 + _col]
#           self.mask[(act_len * _row):(act_len* (_row + 1)), (400 + act_len * _col):(400 + act_len * (_col + 1)), :3] = act_to_show   
#       # For the last row
#       if(col_last > 0):
#         for _col in range(col_last):
#           act_to_show = resized_activations[row_num * 10 + _col]
#           self.mask[(act_len * row_num):(act_len* (row_num + 1)), (400 + act_len * _col):(400 + act_len * (_col + 1)), :3] = act_to_show   
#     # If there is more than 100 activations to show
#     else:
#       for _row in range(10):
#         for _col in range(10):
#           act_to_show = resized_activations[_row * 10 + _col]
#           self.mask[(act_len * _row):(act_len* (_row + 1)), (400 + act_len * _col):(400 + act_len * (_col + 1)), :3] = act_to_show   
#     # Adding the prediction label and selected layer 
#     self.showImgText(predictions, selected_layer)
#     cv2.imshow('XAI_tool', self.mask)
#     cv2.waitKey(100)