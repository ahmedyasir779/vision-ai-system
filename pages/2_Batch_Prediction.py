import streamlit as st
import torch
from pathlib import Path
import sys
from PIL import Image
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cnn_model import ModelFactory
from src.inference import InferenceEngine
import cv2
import numpy as np

st.set_page_config(
    page_title="Batch Prediction",
    page_icon="🔮",
    layout="wide"
)

st.title("🔮 Batch Prediction")
st.markdown("Upload multiple images and predict them all at once!")
st.markdown("---")


@st.cache_resource
def load_model():
    """Load trained model"""
    checkpoint_path = 'models/checkpoints_real/best_model.pth'
    class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model = ModelFactory.create_model(
        'resnet18',
        num_classes=len(class_names),
        pretrained=False
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, class_names


# Load model
model, class_names = load_model()

# File uploader for multiple files
uploaded_files = st.file_uploader(
    "Choose images...",
    type=['jpg', 'jpeg', 'png'],
    accept_multiple_files=True,
    help="Upload multiple images for batch prediction"
)

if uploaded_files:
    st.success(f"✅ {len(uploaded_files)} images uploaded!")
    
    if st.button("🚀 Predict All", type="primary", use_container_width=True):
        
        # Create inference engine
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        engine = InferenceEngine(
            model=model,
            class_names=class_names,
            device=device
        )
        
        results = []
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Create temporary directory
        temp_dir = Path("uploads/batch")
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        for idx, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Processing {idx+1}/{len(uploaded_files)}: {uploaded_file.name}")
            
            # Save temporarily
            image = Image.open(uploaded_file)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            img_array = np.array(image)
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            temp_path = temp_dir / f"temp_{idx}.jpg"
            cv2.imwrite(str(temp_path), img_array)
            
            # Predict
            result = engine.predict_single(str(temp_path))
            
            results.append({
                'Image': uploaded_file.name,
                'Prediction': result['predicted_class'],
                'Confidence': f"{result['confidence']*100:.2f}%",
                'Top 2': result['top3_classes'][1],
                'Top 3': result['top3_classes'][2]
            })
            
            progress_bar.progress((idx + 1) / len(uploaded_files))
        
        status_text.text("✅ All predictions complete!")
        
        # Display results
        st.markdown("---")
        st.subheader("📊 Batch Results")
        
        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True)
        
        # Download results
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results CSV",
            data=csv,
            file_name="batch_predictions.csv",
            mime="text/csv"
        )
        
        # Show image grid with predictions
        st.markdown("---")
        st.subheader("🖼️ Image Grid")
        
        cols = st.columns(3)
        
        for idx, uploaded_file in enumerate(uploaded_files):
            col_idx = idx % 3
            
            with cols[col_idx]:
                image = Image.open(uploaded_file)
                st.image(image, use_column_width=True)
                
                result = results[idx]
                st.markdown(f"**{result['Prediction']}**")
                st.caption(f"Confidence: {result['Confidence']}")
                st.markdown("---")

else:
    st.info("👆 Upload multiple images to get started!")