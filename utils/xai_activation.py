from keras import models
import numpy as np

class XAIActivations:
    def __init__(self, img_tensor, model, layers):
        self.img_tensor = img_tensor
        self.model = model
        self.layers = layers

    def turnTo3d(self, activations_list):
        activations_3d_list = []
        for activation in activations_list:
            activation_3d = []
            for c_channel in range(3):
                activation_3d.append(activation)
            activation_3d = np.array(activation_3d)
            activation_3d = np.transpose(activation_3d)
            activations_3d_list.append(activation_3d)
        return activations_3d_list


    def runTool(self, layer_num):
        activation_list = self.visualise_intermediate_layers(self.img_tensor, self.model, self.layers[layer_num])
        activation_3d_list = self.turnTo3d(activation_list)
        return activation_3d_list
        

    def visualise_intermediate_layers(self, image, keras_model, desired_layer, channel_to_display = None):
        # Creating the model based on the layer selected
        layer_outputs = [keras_model.get_layer(desired_layer).output]
        activation_model = models.Model(inputs = keras_model.input, outputs = layer_outputs)
        
        # Get all the activations from the desired layer
        layer_activations = activation_model.predict(image)
        
        # This is done such that the (1, size, size, num_features) layer activations np array just becomes (size, size, num_features)
        # Since there would only be one layer
        layer_activations = layer_activations[0]
        
        # List of channel_images to return
        channel_image_list = []
        
        # Number of features in the feature map of that one layer
        num_features = layer_activations.shape[-1]
            
        # Now, we need to loop through all the channels/features of that one desired layer
        for channel_index in range(num_features):
            channel_image = layer_activations[:, :, channel_index]
            channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            channel_image_list.append(channel_image)


        # If a channel is selected to be displayed, just return a list with one channel image
        if(channel_to_display or channel_to_display == 0):
            return [channel_image_list[channel_to_display]]
        # Else just return the whole list
        else:
            return channel_image_list