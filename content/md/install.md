# Set-Up

This section covers more on how to adjust the code to allow you run the tool through your own model and images.

We assume you have followed the `readme.md` instructions on how to get the basic set-up working. If you have not, click [here](https://github.com/chuanhao01/XVis_Tool/blob/master/README.md)  

### A few pointers before adjusting the code

The main way you will be adjusting the template code to fit your model is through the input parameters into the class XVisTool, as well as your global variables used in the code.  

The XVisTool is defined as:
```python
class XVisTool:
    def __init__(self, model, input_size, decoder_func = None, preprocess_img_func = None):
```

As you can see, it takes in a total of 4 arguments with 2 of them being optional.

Other than the arguments you see above there is also:  
```python
img_path = 'path_to_img' # Not needed for multi threaded implementation
target_labels = []
```

Thus in total we have around 5-6 variables to change to allow the template code to be working on other models.

These are: 
```python
model = keras_model
img_path = 'path_to_img' # Not needed for multi threaded implementation
input_size = (size_dim, size_dim)
target_label = ['pred_if_output_is_1', ...] # Optional, can be None 
preprocess_input = def a_function # Optional can also be None
```

Note; target_label are the labels your model is trying to predict, with the same order as it predictions. Also you are only required to give the target labels as there is already a wrapper function to help you create the final decoder function used. If that gives you an error, comment out the wrapper function and write your own. It should stored in the decode variable.

### Adjusting the Single threaded template

This will be a guide on how to adjust `single.py` so that it uses your model and test image instead.

Looking at a snippet of the code:  
```python
# Setting up the input for the tool
# model is a keras model
# input_size is a tuple of (x, y) which contains the dims of the input image of the CNN
# decode_predictions is the function keras uses to decode the prediction
# preprocess_input is the function keras uses to process the images before passing them into the model. NOTE: this is an optional arg

from keras.applications.xception import Xception, decode_predictions, preprocess_input
from keras.models import load_model

# Changes come below here 

model = Xception(weights = 'imagenet')
# model = load_model('models/mobilenet_v2_2_acc.hdf5')
img_path = 'Sample_Images/doberman_1.jpg'
# model = load_model('models/best_xception_based_mnist.h5')
input_size = (299, 299) 
target_labels = None
# target_labels = [str(i) for i in range(10)]
preprocess_input = preprocess_input 

# Wrapper so as to automatically create the decoder function
def createDecoder(target_labels):
    if(target_labels is not None):
        def decode(model, img_tensor):
            preds = model.predict(img_tensor)
            # target_labels = target_labels
            pred_index = preds.argmax(axis=-1)[0]
            return [target_labels[pred_index], preds[0][pred_index]]    
    else:
        def decode(model, img_tensor):
            preds = model.predict(img_tensor)
            preds = decode_predictions(preds, top=1)[0][0]
            preds = [preds[1], preds[2]]
            return preds
    return decode

# Custom preprocess input function if you have one
# def preprocess_input(img_tensor):
#         img_tensor = img_tensor / 255 
#         return img_tensor

decode = createDecoder(target_labels)

```
Step 1:  
To make it use a custom model to predict, comment out the line that loads in the pre-trained xception model and uncomment the line under to load in your own model. Then add the path to your model in the string.  

Step 2:  
Then change the img_path to a string that points to where the img you wanted loaded.

Step 3:
Change the target label so that it is a list of your predictions

(Optional)
Step 4:
Change the decode function and preprocess_input function if necessary.

### Adjusting the multi threaded template

This will be a guide on how to adjust `multi.py` so that it uses your model and test image instead.

Looking at a snippet of the code:  
```python
def xaiProcessing():
    # Need to import the keras mdoel in the thread itself
    from keras.applications.xception import Xception, decode_predictions, preprocess_input
    from keras.models import load_model
    model = Xception(weights = 'imagenet')
    # model = load_model('models/mobilenet_v2_2_acc.hdf5')
    input_size = (299, 299) 
    target_labels = None
    # target_labels = ['ChairPose', 'ChestBump', 'ChildPose', 'Dabbing', 'EaglePose', 'HandGun', 'HandShake', 'HighKneel', 'HulkSmash', 'KoreanHeart', 'KungfuCrane', 'KungfuSalute', 'Salute', 'Spiderman', 'WarriorPose']
    preprocess_input = preprocess_input 

    # Wrapper so as to automatically create the decoder function
    def createDecoder(target_labels):
        if(target_labels is not None):
            def decode(model, img_tensor):
                preds = model.predict(img_tensor)
                # target_labels = target_labels
                pred_index = preds.argmax(axis=-1)[0]
                return [target_labels[pred_index], preds[0][pred_index]]    
        else:
            def decode(model, img_tensor):
                preds = model.predict(img_tensor)
                preds = decode_predictions(preds, top=1)[0][0]
                preds = [preds[1], preds[2]]
                return preds
        return decode
    
    # Preprocess input function goes here
    # def preprocess_input(img_tensor):
    #     img_tensor = img_tensor / 255 
    #     return img_tensor

    decode = createDecoder(target_labels)
```

The steps to take to change this are slightly different, as there is no need for an img_path.

Things to note:
1. The code to change is inside a function as this function is running in its own thread. Be careful when declaring variables or importing libraries in this case.
2. The video/frames are being taken from the webcam, and is default to `cv2.VideoCapture(0)`. If you are using other webcams or would like to do video integration, please look at cv2's documentation.

