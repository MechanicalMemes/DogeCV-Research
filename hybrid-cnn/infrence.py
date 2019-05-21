from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

import cv2
from keras.models import load_model



model = load_model("model.hdf5")
print(model.summary())
output_names = [node.op.name for node in model.inputs]

print(output_names)
def predict(file):
    img = load_img(file)  # this is a PIL image
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x = cv2.resize(x, (64,64))

    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

    classes = ['gold', 'none', 'silver']
    #print(model.predict(x))
    res = model.predict_classes(x)
    return classes[res[0]]

print(predict("test-1.jpg"))
print(predict("test-2.jpg"))
print(predict("test-3.jpg"))
print(predict("test-4.jpg"))
#print(predict("test-5.jpg"))