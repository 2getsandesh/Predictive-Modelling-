import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image

# Load the saved model
model = tf.keras.models.load_model('plant_type_classifier2.h5')

# Define image dimensions
img_width, img_height = 150, 150

# Define the list of plant class names
class_labels = [
    "aloevera", "banana", "bilimbi", "cantaloupe", "cassava", "coconut", "corn", "cucumber",
    "curcuma", "eggplant", "galangal", "ginger", "guava", "kale", "longbeans", "mango", "melon",
    "orange", "paddy", "papaya", "peper chili", "pineapple", "pomelo", "shallot", "soybeans",
    "spinach", "sweet potatoes", "tobacco", "waterapple", "watermelon"
]

st.title("Plant Type Classifier")

# Allow the user to upload an image
uploaded_image = st.file_uploader("Upload an image...", type=[
                                  "jpg", "png", "JPEG", "jpeg", "JPG", "PNG"])

if uploaded_image is not None:
    # Preprocess the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    image = image.resize((img_width, img_height))
    img = img_to_array(image)
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Make predictions
    prediction = model.predict(img)

    # Get the top 5 plant types with the highest confidence
    top_indices = np.argsort(prediction[0])[::-1][:5]
    top_confidences = [prediction[0][i] for i in top_indices]
    top_classes = [class_labels[i] for i in top_indices]

    st.subheader("Top 5 Predictions:")

    for i in range(5):
        st.write(f"{i + 1}. {top_classes[i]} ({top_confidences[i]:.2%})")
