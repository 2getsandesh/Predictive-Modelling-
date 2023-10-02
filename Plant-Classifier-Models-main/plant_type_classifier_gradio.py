import gradio as gr
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# Load the saved model
model = tf.keras.models.load_model('plant_type_classifier.h5')

# Define image dimensions
img_width, img_height = 150, 150

# Define the list of plant class names
class_labels = [
    "aloevera", "banana", "bilimbi", "cantaloupe", "cassava", "coconut", "corn", "cucumber",
    "curcuma", "eggplant", "galangal", "ginger", "guava", "kale", "longbeans", "mango", "melon",
    "orange", "paddy", "papaya", "peper chili", "pineapple", "pomelo", "shallot", "soybeans",
    "spinach", "sweet potatoes", "tobacco", "waterapple", "watermelon"
]

# Define a Gradio interface


def classify_plant(image):
    # Preprocess the image
    image = Image.fromarray((image * 255).astype('uint8'))
    image = image.resize((img_width, img_height))
    img = img_to_array(image)
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Make predictions
    prediction = model.predict(img)

    # Get the top plant type with the highest confidence
    top_index = np.argmax(prediction)
    top_class = class_labels[top_index]
    top_confidence = prediction[0][top_index]

    return top_class


# Create a Gradio interface
iface = gr.Interface(
    fn=classify_plant,
    inputs=gr.Image(shape=(img_width, img_height)),
    outputs=gr.Label(),
    live=True
)

iface.launch()
