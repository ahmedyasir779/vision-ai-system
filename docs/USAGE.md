# 📖 Usage Guide

Complete guide for using the Vision AI System.

---

## 🚀 Getting Started

### 1. Installation

Choose one of these methods:

#### Docker (Easiest)
```bash
docker pull ahmedyasir779/vision-ai-system:latest
docker run -p 8501:8501 ahmedyasir779/vision-ai-system:latest
```

#### Local Installation
```bash
git clone https://github.com/ahmedyasir779/vision-ai-system.git
cd vision-ai-system
pip install -r requirements.txt
streamlit run app.py
```

---

## 🎨 Using the Web Interface

### Main Page - Single Prediction

1. **Upload Image**: Click "Choose an image..." and select a flower image
2. **Predict**: Click the "🔮 Predict" button
3. **View Results**: See prediction, confidence, and probability distribution

**Supported formats**: JPG, JPEG, PNG

### Performance Dashboard

Navigate to **📊 Model Performance** to view:
- Training & validation curves
- Confusion matrix
- Classification report
- Downloadable metrics

### Batch Prediction

Navigate to **🔮 Batch Prediction** to:
1. Upload multiple images
2. Click "🚀 Predict All"
3. View results table
4. Download results as CSV

---

## 💻 Using Python API

### Basic Prediction
```python
from src.cnn_model import ModelFactory
from src.inference import InferenceEngine
import torch

# Load model
model = ModelFactory.create_model('resnet18', num_classes=5)
checkpoint = torch.load('models/checkpoints_real/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Create inference engine
engine = InferenceEngine(
    model=model,
    class_names=['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips'],
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Predict
result = engine.predict_single('path/to/image.jpg')
print(f"Prediction: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### Batch Prediction
```python
image_paths = ['image1.jpg', 'image2.jpg', 'image3.jpg']
results = engine.predict_batch(image_paths)

for result in results:
    print(f"{result['image_path']}: {result['predicted_class']}")
```

---

## 🎓 Training Custom Model

### 1. Prepare Dataset
```python
from src.data_downloader_real import DatasetPreparer

preparer = DatasetPreparer()
dataset_dir = preparer.prepare_flowers_dataset(samples_per_class=150)
```

### 2. Split Dataset
```python
from src.dataset_organizer import DatasetOrganizer

organizer = DatasetOrganizer(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
organizer.split_dataset(dataset_dir, 'data/processed_custom')
```

### 3. Train Model
```python
from src.dataset_loader import DataLoaderFactory
from src.cnn_model import ModelFactory
from src.trainer import Trainer
import torch.nn as nn
import torch.optim as optim

# Create dataloaders
factory = DataLoaderFactory('data/processed_custom', batch_size=16)
train_loader, val_loader, test_loader, class_to_idx = factory.create_dataloaders()

# Create model
model = ModelFactory.create_model('resnet18', num_classes=len(class_to_idx))

# Setup training
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train
trainer = Trainer(model, train_loader, val_loader, criterion, optimizer)
history = trainer.train(num_epochs=10)
```

---

## 🔧 Configuration

### Model Selection

Edit `app.py` to change default model:
```python
model_type = st.selectbox(
    "Select Model",
    ["real", "synthetic", "custom"]  # Add your model
)
```

### Augmentation

Modify `src/augmentation.py`:
```python
# Change augmentation parameters
self.train_transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    # Add your augmentations
])
```

### Training Parameters

Edit training script:
```python
trainer = Trainer(
    model=model,
    # ... other params
    checkpoint_dir='models/my_custom_model'  # Custom checkpoint dir
)
```

---

## 📊 Interpreting Results

### Confidence Scores

- **>90%**: Very confident prediction
- **70-90%**: Good confidence
- **50-70%**: Moderate confidence
- **<50%**: Low confidence (may be uncertain)

### Confusion Matrix

- **Diagonal**: Correct predictions
- **Off-diagonal**: Misclassifications
- **Darker colors**: Higher values

---

## ❓ Troubleshooting

### Model not found
```bash
# Download or train model first
python tests/test_day46.py
```

### CUDA out of memory
```python
# Reduce batch size in config
batch_size = 8  # Instead of 16
```

### Poor predictions
- Check image quality
- Ensure image is of supported class
- Retrain with more data

---

## 📞 Support

- 🐛 [Report Bug](https://github.com/ahmedyasir779/vision-ai-system/issues)
- 💡 [Request Feature](https://github.com/ahmedyasir779/vision-ai-system/issues)
- 📧 Contact: [LinkedIn](https://www.linkedin.com/in/ahmed-yasir-907561206)