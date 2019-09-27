# XVis Tool

### Going through the code

Looking at the class from the start, we can see that the class takes in 4 arguments, with 2 being optional.  
```python
class XVisTool:
    def __init__(self, model, input_size, decoder_func = None, preprocess_img_func = None):
```
In this case, we have taken into account that most models have their own preprocessing and decoder function.

Note: The presence of a preprocessing function and decoder function will affect how the class behaves.

From here you will be either doing single or multi threaded processing, thus the methods called will be slightly different.

**For the single threaded approach:**  
The class assumes you will not be changing the input image and thus the input image is set as a class variable with the `setStillImage` method.
```python
    # For still img takes in the path and loads it into the class, then returns the cv2 imread of the image
    def setStillImg(self, img_path):
```

Then you will call the wrapper method `stillImgRun` with your desired layer to let the class process your image into a 4-d array for batch input into the keras model. The method then would use that image tensor to generate the heatmap and activations using the `xai_heatmap` and `xai_activations` classes. Lastly, if there is a decoding function present, it will also decode the prediction that the model gives into a label and is returned.

```python
    # High level wrapper for the still img function
    # Note: I'm sure there is an even better way to implement this but I leaving this as it is
    def stillImgRun(self, selected_layer):
```

**For the multi threaded approach:**  

As the frame that you want to generate the heatmap and activations for will be constantly updated, we will be passing in that frame into the class method everytime we call it. Besides that there is not much different when calling this class method, `vidCapRun`.

```python
    # High level wrapper for the function, return the heatmap and activation in a dic
    # Will also return preds if present
    def vidCapRun(self, frame, selected_layer):
```