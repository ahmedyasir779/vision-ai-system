import streamlit as st

st.set_page_config(
    page_title="About",
    page_icon="📚",
    layout="wide"
)

st.title("📚 About Vision AI System")
st.markdown("---")

# Project info
col1, col2 = st.columns([2, 1])

with col1:
    st.header("🤖 Project Overview")
    st.markdown("""
    **Vision AI System** is a production-ready image classification system built with:
    
    - **Deep Learning**: ResNet18 architecture with transfer learning
    - **Real Data**: Trained on 750+ real flower images
    - **High Accuracy**: 85-95% validation accuracy
    - **Interactive UI**: Beautiful Streamlit interface
    - **Fast Inference**: GPU-accelerated predictions
    
    ### 🎯 Features
    
    - ✅ Single image prediction with confidence scores
    - ✅ Batch prediction for multiple images
    - ✅ Real-time visualization of results
    - ✅ Performance monitoring dashboard
    - ✅ Model comparison (synthetic vs real data)
    - ✅ Downloadable results (CSV format)
    
    ### 📊 Dataset
    
    **Flowers Classification Dataset**
    - 5 Categories: Daisy, Dandelion, Roses, Sunflowers, Tulips
    - ~750 total images
    - Split: 70% train, 15% validation, 15% test
    - Resolution: 224x224 pixels
    
    ### 🏗️ Architecture
    
    **ResNet18** (Residual Neural Network)
    - Pre-trained on ImageNet
    - Fine-tuned on flowers dataset
    - ~11M parameters
    - Transfer learning approach
    
    ### 🎓 Training Details
    
    - **Epochs**: 10
    - **Batch Size**: 16
    - **Optimizer**: Adam (lr=0.001)
    - **Loss**: CrossEntropyLoss
    - **Augmentation**: Flip, rotate, brightness, contrast
    """)

with col2:
    st.header("👨‍💻 Developer")
    
    st.image("https://via.placeholder.com/300x300.png?text=Ahmed+Yasir", use_column_width=True)
    
    st.markdown("""
    **Ahmed Yasir (Dee)**
    
    AI/ML Engineer in Training
    
    🌍 Riyadh, Saudi Arabia
    
    ### 🔗 Connect
    
    - [GitHub](https://github.com/ahmedyasir779)
    - [LinkedIn](https://www.linkedin.com/in/ahmed-yasir-907561206)
    - [Discord](https://discord.gg/8JhdzgzP)
    
    ### 🎯 2024 Goals
    
    - ✅ Build 12 production AI projects
    - ✅ Contribute to 10+ open source repos
    - 🔄 Publish 2 models to Hugging Face
    - 🔄 Write 50+ technical articles
    """)

st.markdown("---")

# Tech stack
st.header("🛠️ Tech Stack")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🤖 Deep Learning")
    st.markdown("""
    - PyTorch 2.1.0
    - TorchVision 0.16.0
    - Albumentations 1.3.1
    - Scikit-learn 1.3.2
    """)

with col2:
    st.subheader("📊 Data & Visualization")
    st.markdown("""
    - NumPy 1.26.2
    - Pandas 2.1.4
    - Matplotlib 3.8.2
    - Plotly 5.18.0
    - Seaborn 0.13.0
    """)

with col3:
    st.subheader("🌐 Web & Deployment")
    st.markdown("""
    - Streamlit 1.29.0
    - OpenCV 4.8.1
    - Pillow 10.1.0
    - Docker (ready)
    """)

st.markdown("---")

# Timeline
st.header("📅 Development Timeline")

timeline_data = {
    "Day": [43, 44, 45, 46, 47],
    "Task": [
        "Image Processing Fundamentals",
        "Dataset Preparation & Augmentation",
        "CNN Architecture & Training",
        "Real Data Training & Evaluation",
        "Streamlit UI & Demo"
    ],
    "Status": ["✅", "✅", "✅", "✅", "✅"]
}

import pandas as pd
st.table(pd.DataFrame(timeline_data))

st.markdown("---")

# Footer
st.markdown("""
<div style="text-align: center; color: #888;">
    <p>Vision AI System v1.0</p>
    <p>© 2024 Ahmed Yasir | Building in Public 🚀</p>
</div>
""", unsafe_allow_html=True)