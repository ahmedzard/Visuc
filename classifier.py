# Loading the model that we have trained

## Importing Libraries
import numpy as np

# Importing Tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#print(tf.__version__)

"""## Loading Trained Model"""
# Recreate the exact same model, including its weights and the optimizer
model = tf.keras.models.load_model(r'C:\Users\be174.BARQSYSTEMS\Desktop\Artathoen\sample_data\colormodel_trained_90.h5')

"""## Initializing Color Classes for Prediction"""

# Mapping the Color Index with the respective 11 Classes (More Explained in RGB Color Classifier: Part 1)
color_dict={
    0 : 'Red',
    1 : 'Green',
    2 : 'Blue',
    3 : 'Yellow',
    4 : 'Orange',
    5 : 'Pink',
    6 : 'Purple',
    7 : 'Brown',
    8 : 'Grey',
    9 : 'Black',
    10 : 'White'
}

#predicting from loaded trained_model
def predict_color(Red, Green, Blue):
    rgb = np.asarray((Red, Green, Blue)) #rgb tuple to numpy array
    input_rgb = np.reshape(rgb, (-1,3)) #reshaping as per input to ANN model
    color_class_confidence = model.predict(input_rgb) # Output of layer is in terms of Confidence of the 11 classes
    color_index = np.argmax(color_class_confidence, axis=1) #finding the color_class index from confidence
    color = color_dict[int(color_index)]
    return color


color_predicted = predict_color(213,5,150)


# Change depending on image name
from PIL import Image
import os
image_path=os.path.abspath(r"C:\Users\be174.BARQSYSTEMS\Desktop\Artathoen\sample_data\white1.png")
img = Image.open(image_path)
img.show()

### First Thing Get RGB Value of Uploaded Image, then use the rgb values and pass them to your classifer of colors which takes rgb values and predict color
# Importing Image from PIL package
from PIL import Image

def get_rgb_value_of_pic(image_path):
  img = Image.open(image_path)
  #img.show()
  rgb_im = img.convert('RGB')
  r, g, b = rgb_im.getpixel((1, 1))
  #coordinate = x, y = 100, 100
  #rgb_values = img.getpixel(coordinate)
  predicted_color = predict_color(r,g,b)
  return predicted_color

predicted_color = get_rgb_value_of_pic(image_path)
print(predicted_color)



# Secondly pass the predicted colro to music fn.
from playsound import playsound
def music_fn(predicted_color):
  if predicted_color =='Red':
    playsound(r'C:\Users\be174.BARQSYSTEMS\Desktop\Artathoen\sample_data\music1_wav.wav')
    print('music1')
  elif predicted_color =='Green':
    playsound(r'C:\Users\be174.BARQSYSTEMS\Desktop\Artathoen\sample_data\music2_wav.wav')
    print('music2')
  elif predicted_color =='Blue':
    playsound(r'C:\Users\be174.BARQSYSTEMS\Desktop\Artathoen\sample_data\music3_wav.wav')
    print('music3')
  elif predicted_color =='Yellow':
    playsound(r'C:\Users\be174.BARQSYSTEMS\Desktop\Artathoen\sample_data\music4_wav.wav')
    print('music4')
  elif predicted_color =='Orange':
    playsound(r'C:\Users\be174.BARQSYSTEMS\Desktop\Artathoen\sample_data\music5_wav.wav')
    print('music5')	
  elif predicted_color =='Pink':
    playsound(r'C:\Users\be174.BARQSYSTEMS\Desktop\Artathoen\sample_data\music6_wav.wav')
    print('music6')	
  elif predicted_color =='Purple':
    playsound(r'C:\Users\be174.BARQSYSTEMS\Desktop\Artathoen\sample_data\music7_wav.wav')
    print('music7')	
  elif predicted_color =='Brown':
    playsound(r'C:\Users\be174.BARQSYSTEMS\Desktop\Artathoen\sample_data\music8_wav.wav')
    print('music8')	
  elif predicted_color =='Gray':
    playsound(r'C:\Users\be174.BARQSYSTEMS\Desktop\Artathoen\sample_data\music9_wav.wav')
    print('music9')	
  elif predicted_color =='Black':
    playsound(r'C:\Users\be174.BARQSYSTEMS\Desktop\Artathoen\sample_data\music10_wav.wav')
    print('music10')
  else:
    playsound(r'C:\Users\be174.BARQSYSTEMS\Desktop\Artathoen\sample_data\music11_wav.wav')
    print('music9')

print(music_fn(predicted_color))