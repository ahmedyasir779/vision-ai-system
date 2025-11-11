import streamlit as st
import json
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import pandas as pd

st.set_page_config(
    page_title="Model Performance",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Model Performance Dashboard")
st.markdown("---")

# Load training history
try:
    with open('logs/training_history_real.json', 'r') as f:
        history = json.load(f)
    
    # Convert to DataFrame
    df = pd.DataFrame({
        'Epoch': range(1, len(history['train_loss']) + 1),
        'Train Loss': history['train_loss'],
        'Val Loss': history['val_loss'],
        'Train Accuracy': history['train_acc'],
        'Val Accuracy': history['val_acc']
    })
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Best Val Accuracy",
            f"{max(history['val_acc']):.2f}%",
            delta=f"+{max(history['val_acc']) - history['val_acc'][0]:.2f}%"
        )
    
    with col2:
        st.metric(
            "Final Train Loss",
            f"{history['train_loss'][-1]:.4f}",
            delta=f"{history['train_loss'][-1] - history['train_loss'][0]:.4f}"
        )
    
    with col3:
        st.metric(
            "Final Val Loss",
            f"{history['val_loss'][-1]:.4f}",
            delta=f"{history['val_loss'][-1] - history['val_loss'][0]:.4f}"
        )
    
    with col4:
        st.metric(
            "Total Epochs",
            len(history['train_loss'])
        )
    
    st.markdown("---")
    
    # Training curves
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📉 Loss Curves")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['Epoch'],
            y=df['Train Loss'],
            mode='lines+markers',
            name='Train Loss',
            line=dict(color='#FF6B6B', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=df['Epoch'],
            y=df['Val Loss'],
            mode='lines+markers',
            name='Val Loss',
            line=dict(color='#4ECDC4', width=3)
        ))
        
        fig.update_layout(
            xaxis_title="Epoch",
            yaxis_title="Loss",
            template='plotly_dark',
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Accuracy Curves")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['Epoch'],
            y=df['Train Accuracy'],
            mode='lines+markers',
            name='Train Accuracy',
            line=dict(color='#FF6B6B', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=df['Epoch'],
            y=df['Val Accuracy'],
            mode='lines+markers',
            name='Val Accuracy',
            line=dict(color='#4ECDC4', width=3)
        ))
        
        fig.update_layout(
            xaxis_title="Epoch",
            yaxis_title="Accuracy (%)",
            template='plotly_dark',
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Training data table
    st.subheader("📋 Training History")
    st.dataframe(df, use_container_width=True)
    
    # Download button
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name="training_history.csv",
        mime="text/csv"
    )

except FileNotFoundError:
    st.error("❌ Training history not found! Please train the model first.")
except Exception as e:
    st.error(f"❌ Error loading training history: {e}")

# Show confusion matrix
st.markdown("---")
st.subheader("🎯 Confusion Matrix")

confusion_matrix_path = Path("results/plots/confusion_matrix_real.png")

if confusion_matrix_path.exists():
    st.image(str(confusion_matrix_path), use_column_width=True)
else:
    st.warning("⚠️ Confusion matrix not found!")

# Show classification report
st.markdown("---")
st.subheader("📄 Classification Report")

report_path = Path("results/classification_report_real.txt")

if report_path.exists():
    with open(report_path, 'r') as f:
        report = f.read()
    st.text(report)
else:
    st.warning("⚠️ Classification report not found!")