import streamlit as st
import torch
import torch.nn as nn
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
import json
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cnn_model import ModelFactory
from src.inference import InferenceEngine

# Page config
st.set_page_config(
    page_title="Vision AI System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    h1 {
        color: #FF4B4B;
        padding-bottom: 1rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #262730;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(model_type: str = 'real'):
    """Load trained model"""
    try:
        if model_type == 'real':
            checkpoint_path = 'models/checkpoints_real/best_model.pth'
            class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
        else:
            checkpoint_path = 'models/checkpoints/best_model.pth'
            class_names = ['bird', 'car', 'cat', 'dog', 'flower']
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Create model
        model = ModelFactory.create_model(
            'resnet18',
            num_classes=len(class_names),
            pretrained=False
        )
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model, class_names, checkpoint.get('best_val_acc', 0)
    
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, 0


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocess uploaded image"""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to numpy array
    img_array = np.array(image)
    
    # Convert RGB to BGR for OpenCV
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    return img_array


def plot_predictions(predictions: dict, class_names: list):
    """Create interactive prediction plot"""
    probs = predictions['all_probabilities']
    
    # Create bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=probs,
            y=class_names,
            orientation='h',
            marker=dict(
                color=probs,
                colorscale='Viridis',
                showscale=True
            ),
            text=[f'{p*100:.1f}%' for p in probs],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Class Probabilities",
        xaxis_title="Confidence",
        yaxis_title="Class",
        height=400,
        showlegend=False,
        template='plotly_dark'
    )
    
    return fig


def main():
    """Main app"""
    
    # Header
    st.title("🤖 Vision AI System")
    st.markdown("**Image Classification with Deep Learning**")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        model_type = st.selectbox(
            "Select Model",
            ["real", "synthetic"],
            format_func=lambda x: "Real Data (Flowers)" if x == "real" else "Synthetic Data"
        )
        
        st.markdown("---")
        
        st.header("📊 Model Info")
        
        # Load model
        with st.spinner("Loading model..."):
            model, class_names, val_acc = load_model(model_type)
        
        if model is None:
            st.error("❌ Model not found! Please train the model first.")
            st.stop()
        
        st.success("✅ Model loaded!")
        st.metric("Validation Accuracy", f"{val_acc:.2f}%")
        st.metric("Classes", len(class_names))
        
        with st.expander("🏷️ Class Names"):
            for i, name in enumerate(class_names):
                st.write(f"{i+1}. {name}")
        
        st.markdown("---")
        
        st.header("ℹ️ How to Use")
        st.markdown("""
        1. Upload an image
        2. Click 'Predict'
        3. View results & confidence
        4. Try different images!
        """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image to classify"
        )
        
        if uploaded_file is not None:
            # Display image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            # Predict button
            if st.button("🔮 Predict", type="primary", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    # Preprocess
                    img_array = preprocess_image(image)
                    
                    # Save temporarily
                    temp_path = Path("uploads/temp.jpg")
                    temp_path.parent.mkdir(exist_ok=True)
                    cv2.imwrite(str(temp_path), img_array)
                    
                    # Create inference engine
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                    engine = InferenceEngine(
                        model=model,
                        class_names=class_names,
                        device=device
                    )
                    
                    # Predict
                    result = engine.predict_single(str(temp_path))
                    
                    # Store in session state
                    st.session_state['result'] = result
                    st.session_state['image'] = image
    
    with col2:
        st.header("📊 Results")
        
        if 'result' in st.session_state:
            result = st.session_state['result']
            
            # Main prediction
            st.markdown("""
                <div class="prediction-box">
                    <h2 style="color: #4CAF50; margin: 0;">Prediction</h2>
                    <h1 style="margin: 0.5rem 0;">{}</h1>
                    <p style="font-size: 1.2rem; color: #888;">Confidence: {:.2f}%</p>
                </div>
            """.format(
                result['predicted_class'].upper(),
                result['confidence'] * 100
            ), unsafe_allow_html=True)
            
            # Confidence meter
            st.progress(result['confidence'])
            
            st.markdown("---")
            
            # Top 3 predictions
            st.subheader("🏆 Top 3 Predictions")
            
            for i, (cls, conf) in enumerate(zip(
                result['top3_classes'],
                result['top3_confidences']
            )):
                col_a, col_b, col_c = st.columns([1, 2, 1])
                
                with col_a:
                    st.markdown(f"**#{i+1}**")
                
                with col_b:
                    st.markdown(f"**{cls}**")
                
                with col_c:
                    st.markdown(f"{conf*100:.1f}%")
                
                st.progress(conf)
            
            st.markdown("---")
            
            # Interactive plot
            st.subheader("📈 All Class Probabilities")
            fig = plot_predictions(result, class_names)
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.info("👆 Upload an image and click 'Predict' to see results!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #888;">
            <p>Built with ❤️ by Ahmed Yasir | Vision AI System</p>
            <p>🐙 GitHub: @ahmedyasir779 | 💼 LinkedIn: Ahmed Yasir</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()