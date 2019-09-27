# Drawer

### Going through the code

We would use this class and intialise it only on the highest level. As we can see from its code, it requires no arguments to initialise.  

```python
class Drawer:
  def __init__(self):
    self.mask = np.zeros((800, 1200, 3), dtype='uint8')
```

However the main thing we have to look out for when we use this class is handling its mask variable. As this will be the frame we will be rendering on the cv2 terminal to show to the user.

Again there will be slight differences in how you approach the class depending on if you are looking for a multi threaded approach or a single threaded approach.  

**For single threaded:**  

As you will have the same rendered original image, you will be rendering the original image, heatmap, activations and the selcted layer everytime you change the selected layer to visualize. (Predictions as well is you have them) You do this by calling the `singleThread` method in the drawer function.
```python
  # High level wrapper used for the template Single thread implementation
  def singleThread(self, ori_img, heatmap, activations, select_layers_list, preds = None):
```

Note: You can see here that the input of predcitions is optional.

**For the multi threaded approach:**  

As the frame you would like to render will be updated more quickly than the computer's ability to generate the heatmap and activations, the usage of the drawer class is slight different as well.

In this case, you would use the `drawOriPic` method to update the frame constantly being shown on the terminal.

Then in another thread that is generating the heatmap and activations, you will call the `multiThread` method to help you draw the heatmap and activations on the mask for you.

```python
  # High level wrapper used for the template multi threaded webcam implementation
  def multiThread(self, heatmap, activations,  select_layers_list, preds = None):
```

Note: For the multi threaded approach, you will have to decide when you want to clear the mask to strat anew, as these methods merely override the current mask to draw what you want.