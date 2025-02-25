import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
from PIL import Image
import json
import io
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import gdown
import os
from tensorflow.keras.models import load_model # type: ignore

file_id = "1EXtHrivj3G96mjfoVsPpslxfoHeQbARA"


# Set page configuration
st.set_page_config(
    page_title="AI Potato Disease Detection",
    page_icon="🥔",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS with dark theme support
st.markdown("""
    <style>
    /* Dark theme variables */
    :root {
        --main-bg-color: #0e1117;
        --card-bg-color: #1a1c24;
        --text-color: #e0e0e0;
        --primary-color: #17c3b2;
        --secondary-color: #1e3d59;
        --border-color: #2d3139;
        --hover-color: #14a89a;
    }

    /* Main container styling */
    .main {
        background-color: var(--main-bg-color);
        color: var(--text-color);
    }
    
    /* Streamlit default background override */
    .stApp {
        background-color: var(--main-bg-color);
    }
    
    /* Headers */
    h1 {
        color: var(--text-color) !important;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        text-align: center;
        margin-bottom: 2rem !important;
    }
    
    h2 {
        color: var(--text-color) !important;
        font-size: 1.8rem !important;
        font-weight: 600 !important;
        margin-top: 1.5rem !important;
    }
    
    /* Cards */
    .stCard {
        background-color: var(--card-bg-color);
        color: var(--text-color);
        padding: 2rem;
        border-radius: 1rem;
        border: 1px solid var(--border-color);
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    }
    
    /* Glassmorphism effect */
    .glass-card {
        background: rgba(26, 28, 36, 0.7);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Upload box */
    .uploadBox {
        border: 2px dashed var(--primary-color);
        border-radius: 1rem;
        padding: 2rem;
        text-align: center;
        margin: 2rem 0;
        background-color: var(--card-bg-color);
    }
    
    /* Metrics styling */
    .metric-container {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        margin: 2rem 0;
    }
    
    .metric-box {
        background-color: var(--card-bg-color);
        padding: 1.5rem;
        border-radius: 1rem;
        border: 1px solid var(--border-color);
        flex: 1;
        text-align: center;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--primary-color);
    }
    
    .metric-label {
        font-size: 1rem;
        color: var(--text-color);
        margin-top: 0.5rem;
    }
    
    /* Progress bars */
    .stProgress > div > div {
        background-color: var(--primary-color);
    }
    
    /* Buttons */
    .stButton>button {
        background-color: var(--primary-color);
        color: white;
        border-radius: 0.5rem;
        padding: 0.5rem 2rem;
        border: none;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: var(--hover-color);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* Status badges */
    .status-badge {
        padding: 0.5rem 1rem;
        border-radius: 2rem;
        font-weight: 600;
        display: inline-block;
        margin: 0.5rem 0;
    }
    
    .status-healthy {
        background-color: rgba(23, 195, 178, 0.2);
        color: #17c3b2;
        border: 1px solid #17c3b2;
    }
    
    .status-disease {
        background-color: rgba(220, 53, 69, 0.2);
        color: #dc3545;
        border: 1px solid #dc3545;
    }
    
    /* Timeline */
    .timeline {
        margin: 2rem 0;
        padding: 1rem;
        border-left: 2px solid var(--primary-color);
    }
    
    .timeline-item {
        margin-left: 1rem;
        padding: 0.5rem 0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: #666;
        font-size: 0.9rem;
        margin-top: 3rem;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .animate-fade-in {
        animation: fadeIn 0.5s ease-out forwards;
    }
    </style>
""", unsafe_allow_html=True)

# Google Drive file IDs
MODEL_FILE_ID = "1EXtHrivj3G96mjfoVsPpslxfoHeQbARA"
CLASS_INDICES_FILE_ID = "1jUfqkgpQRlmeriFe5vMtbcSfLnLK6dOr"

# File paths
MODEL_PATH = "potato_disease_model.h5"
CLASS_INDICES_PATH = "class_indices.json"

def download_file(file_id, output_path):
    """Download a file from Google Drive if not already present"""
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)

@st.cache_resource
def load_model():
    """Download and load the trained model & class indices"""
    # Download model & class indices if not available
    download_file(MODEL_FILE_ID, MODEL_PATH)
    download_file(CLASS_INDICES_FILE_ID, CLASS_INDICES_PATH)
    
    # Load model
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # Load class indices
    with open(CLASS_INDICES_PATH, 'r') as f:
        class_indices = json.load(f)
    
    # Reverse mapping for labels
    class_labels = {v: k for k, v in class_indices.items()}
    
    return model, class_labels

# Load the model
model, class_labels = load_model()
st.write("✅ Model & class labels loaded successfully!")

def preprocess_image(image):
    """Preprocess the image for model prediction"""
    image = image.resize((224, 224))
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = image_array / 255.0
    return np.expand_dims(image_array, axis=0)

def create_confidence_gauge(confidence, title):
    """Create a gauge chart for confidence visualization"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = confidence * 100,
        title = {'text': title},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "#17c3b2"},
            'steps': [
                {'range': [0, 50], 'color': "rgba(255, 0, 0, 0.2)"},
                {'range': [50, 75], 'color': "rgba(255, 255, 0, 0.2)"},
                {'range': [75, 100], 'color': "rgba(0, 255, 0, 0.2)"}
            ]
        }
    ))
    
    fig.update_layout(
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor = "rgba(0,0,0,0)",
        font = {'color': "#e0e0e0"},
        height = 200
    )
    
    return fig

def create_prediction_history():
    """Create or get prediction history from session state"""
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    return st.session_state.prediction_history

def main():
    # Initialize prediction history
    prediction_history = create_prediction_history()
    
    # Header with animation
    st.markdown("""
        <div class="animate-fade-in">
            <h1>🥔 Advanced AI Potato Disease Detection</h1>
        </div>
    """, unsafe_allow_html=True)
    
    # Introduction card with glass effect
    st.markdown("""
        <div class="stCard glass-card animate-fade-in">
            <h2>AI-Powered Disease Detection</h2>
            <p>Utilizing advanced deep learning technology to provide real-time analysis of potato plant diseases. 
            Our system offers high-accuracy detection with detailed insights and recommendations.</p>
        </div>
    """, unsafe_allow_html=True)
    
    try:
        model, class_labels = load_model()
        
        # Create columns for layout
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<div class="uploadBox animate-fade-in">', unsafe_allow_html=True)
            uploaded_file = st.file_uploader("Upload a potato leaf image", type=["jpg", "jpeg", "png"])
            st.markdown('</div>', unsafe_allow_html=True)
            
            if uploaded_file:
                image = Image.open(uploaded_file)
                st.image(image, caption='Uploaded Image', use_container_width=True)
        
        with col2:
            if uploaded_file:
                with st.spinner('🔬 Analyzing with AI...'):
                    # Process image and get predictions
                    processed_image = preprocess_image(image)
                    predictions = model.predict(processed_image)
                    predicted_class_index = np.argmax(predictions[0])
                    predicted_class = class_labels[predicted_class_index]
                    confidence = float(predictions[0][predicted_class_index])
                    
                    # Add to prediction history
                    prediction_history.append({
                        'timestamp': datetime.now(),
                        'prediction': predicted_class,
                        'confidence': confidence
                    })
                    
                    # Status badge
                    status_class = "status-healthy" if "healthy" in predicted_class.lower() else "status-disease"
                    st.markdown(f"""
                        <div class="status-badge {status_class}">
                            {predicted_class.replace('_', ' ').title()}
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Confidence gauge
                    st.plotly_chart(create_confidence_gauge(confidence, "Confidence Score"), use_container_width=True)
                    
                    # Detailed Analysis Section
                    st.markdown("<h2>Detailed Analysis</h2>", unsafe_allow_html=True)
                    
                    # Create bar chart for all predictions
                    fig = px.bar(
                        x=[class_labels[i].replace('_', ' ').title() for i in range(len(predictions[0]))],
                        y=predictions[0] * 100,
                        labels={'x': 'Disease Category', 'y': 'Confidence (%)'},
                        title='Disease Probability Distribution'
                    )
                    fig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#e0e0e0")
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Recommendations Section
                    st.markdown("<h2>AI Recommendations</h2>", unsafe_allow_html=True)
                    if "healthy" in predicted_class.lower():
                        st.markdown("""
                            <div class="stCard">
                                <h3 style="color: #17c3b2;">✅ Healthy Plant Detected</h3>
                                <div class="timeline">
                                    <div class="timeline-item">▶️ Continue regular maintenance schedule</div>
                                    <div class="timeline-item">▶️ Monitor water and nutrient levels</div>
                                    <div class="timeline-item">▶️ Maintain current environmental conditions</div>
                                    <div class="timeline-item">▶️ Schedule next inspection in 7 days</div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                            <div class="stCard">
                                <h3 style="color: #dc3545;">⚠️ Disease Management Protocol</h3>
                                <div class="timeline">
                                    <div class="timeline-item">🔍 Isolate affected plants immediately</div>
                                    <div class="timeline-item">💊 Apply recommended treatment protocol</div>
                                    <div class="timeline-item">📊 Monitor disease progression daily</div>
                                    <div class="timeline-item">📱 Schedule expert consultation</div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    # Analysis History
                    if len(prediction_history) > 1:
                        st.markdown("<h2>Analysis History</h2>", unsafe_allow_html=True)
                        history_df = pd.DataFrame(prediction_history)
                        fig = px.line(
                            history_df,
                            x='timestamp',
                            y='confidence',
                            title='Confidence Trends Over Time'
                        )
                        fig.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#e0e0e0")
                        )
                        st.plotly_chart(fig, use_container_width=True)
        
        # Footer
        st.markdown("""
            <div class="footer">
                <p>Powered by Advanced Deep Learning | Model Version: 2024.1</p>
                <p>For agricultural research and monitoring purposes only.</p>
            </div>
        """, unsafe_allow_html=True)
                    
    except Exception as e:
        st.error("⚠️ System Error: Please ensure all model files are properly configured.")
        st.info("📮 Need technical support? Contact our AI team.")

if __name__ == '__main__':
    main()
