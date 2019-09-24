import keras
from keras.datasets import mnist

from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions

# Loading MSIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape)

# Init model
model = VGG16(include_top=False, weights=None)
model.summary()
