import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from PIL import Image
import io

# Load the trained model
model_path = "F:/Wound Healing Prediction/Wound/wound_recovery_model.h5"
model = tf.keras.models.load_model(model_path)

# Define class names
class_names = ["No Wound", "Moderate Wound", "Severe Wound"]

# Healing period estimation (days)
healing_periods = {
    "No Wound": "You are alright!",
    "Moderate Wound": "Estimated healing period: 7-14 days",
    "Severe Wound": "Estimated healing period: 15-30 days or requires medical attention"
}

# Function to preprocess the image
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image
    return img_array

# Function to make predictions
def predict_wound(img):
    img_array = preprocess_image(img)
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    predicted_class = class_names[predicted_class_index]
    return predicted_class, healing_periods[predicted_class]

# Streamlit UI
st.sidebar.title("Menu")
page = st.sidebar.selectbox("Select an option:", ["Home", "About Us"])

if page == "Home":
    st.title("Wound Healing Prediction System")
    st.write("Upload an image to check if the wound is affected or not.")
    
    # Image upload
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Convert the file to an image
        img = Image.open(uploaded_file)
        
        # Make prediction
        predicted_class, healing_info = predict_wound(img)
        
        # Display the uploaded image with a fixed size
        st.image(img, caption=f"Prediction: {predicted_class}", width=300)
        
        # Display prediction result
        st.write(f"### {predicted_class}")
        st.write(f"**{healing_info}**")

elif page == "About Us":
    st.title("About Us")
    st.write("""
    ## Welcome to our AI-Based Diagnostic Tool for Wound Healing!
    
    Developed with the power of deep learning, this system helps detect different wound conditions based on image analysis.
    
    ### 🏥 Our Mission
    Our goal is to provide a fast, accurate, and user-friendly solution for wound detection, enabling timely medical intervention and reducing health risks.
    
    ### 🔬 Technology Used
    - **Deep Learning Model:** MobileNetV2
    - **Framework:** TensorFlow & Keras
    - **Web Interface:** Streamlit
    
    ### ⚡ Why It Matters
    - **Easy Detection:** Helps prevent complications.
    - **User-Friendly:** Upload an image and get instant results.
    - **High Accuracy:** Powered by an advanced AI model.
    
    ### 📧 Contact Us
    We'd love to hear from you!
    - **Developer:** P. DhatchinaMoorthy
    - **Email:** datchu15052003@gmail.com
    
    Thank you for exploring our AI-Based Diagnostic Tool!
    
    """)