import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np

# Load the trained model
model = keras.models.load_model('model_inception_.h5')

# Load and preprocess the test image
test_image_path = 'F:\Cotton-Disease-Prediction-Deep-Learning-master/uploads/carrot-leaf-blight_132.jpg'  # Replace with the path to your test image
# Load the image using PIL
image = Image.open(test_image_path)
# Convert the image to RGB format
image = image.convert('RGB')
# Specify the desired input image size for your model
image_size = (224, 224) 
# Resize the image to match the input size of your model
image = image.resize(image_size, resample=Image.BILINEAR)

image_array = np.array(image) / 255.0
# Add an extra dimension to the array to match the model's input shape
input_image = np.expand_dims(image_array, axis=0)

# Make predictions on the test image
predictions = model.predict(input_image)
# Get the predicted class index
predicted_class_index = np.argmax(predictions)
# Get the predicted class label (assuming you hav
# e a list of class labels)
class_labels = ['Black Root Rot','Healthy Carrots','carrot-leaf-blight','carrot-leaf-spot','Root Knot Nematode','White Mold','carrot-dry-leaves','healthy-carrot-leaves','Powder-mildew','Cavity Spot and Root Dieback','carrot-purple-leavesticks','purple Disease','yellow-diseases',]  # Replace with your actual class labels
predicted_class_label = class_labels[predicted_class_index]

# Print the predicted class label
print(f"Predicted class: {predicted_class_label}")