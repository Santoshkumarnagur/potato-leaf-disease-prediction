import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import io

# Set page configuration
st.set_page_config(
    page_title="Potato Disease Detection",
    page_icon="🥔",
    layout="centered"
)

@st.cache_resource
def load_model():
    """Load the trained model and class indices"""
    model = tf.keras.models.load_model('potato_disease_model.h5')
    
    with open('class_indices.json', 'r') as f:
        class_indices = json.load(f)
    
    # Reverse the class indices dictionary
    class_labels = {v: k for k, v in class_indices.items()}
    
    return model, class_labels

def preprocess_image(image):
    """Preprocess the image for model prediction"""
    # Resize image to match training input size
    image = image.resize((224, 224))
    # Convert to array and normalize
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = image_array / 255.0
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Main app
def main():
    # Add custom CSS
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stApp {
            max-width: 800px;
            margin: 0 auto;
        }
        .upload-text {
            text-align: center;
            padding: 2rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # App title and description
    st.title("🥔 Potato Disease Detection")
    st.markdown("""
        Upload an image of a potato leaf to detect if it's healthy or diseased.
        The model will classify the image and provide confidence scores.
    """)
    
    # Load model and class labels
    try:
        model, class_labels = load_model()
        
        # File uploader
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Display progress bar for image processing
            with st.spinner('Processing image...'):
                # Read and display the image
                image = Image.open(uploaded_file)
                st.image(image, caption='Uploaded Image', use_container_width=True)
                
                # Preprocess the image and make prediction
                processed_image = preprocess_image(image)
                predictions = model.predict(processed_image)
                predicted_class_index = np.argmax(predictions[0])
                predicted_class = class_labels[predicted_class_index]
                confidence = float(predictions[0][predicted_class_index])
                
                # Display results
                st.success("Analysis Complete!")
                
                # Create two columns for results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Predicted Disease", predicted_class)
                
                with col2:
                    st.metric("Confidence", f"{confidence:.2%}")
                
                # Display confidence scores for all classes
                st.subheader("Detailed Analysis")
                for idx, conf in enumerate(predictions[0]):
                    disease = class_labels[idx]
                    st.progress(float(conf), text=f"{disease}: {conf:.2%}")
                
                
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Please ensure the model file 'potato_disease_model.h5' and 'class_indices.json' are in the same directory as this script.")

if __name__ == '__main__':
    main()

import gdown
import os

file_id = "149jRjTSqUZS4WhWdgkvFoSnWuEf5OI7V"
url = 'https://drive.google.com/file/d/149jRjTSqUZS4WhWdgkvFoSnWuEf5OI7V/view?usp=sharing'
model_path = "potato_disease_model.h5"


if not os.path.exists(model_path):
    st.warning("Downloading model from Google Drive...")
    gdown.download(url, model_path, quiet=False)