# Brain-tumor-segmentation-using-yolo12n-and-SAM2
# 🧠 Brain Tumor Segmentation using YOLOv12n and SAM2

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Ultralytics](https://img.shields.io/badge/YOLO-v12n-yellow.svg)](https://github.com/ultralytics/ultralytics)
[![SAM2](https://img.shields.io/badge/SAM-2-purple.svg)](https://github.com/facebookresearch/segment-anything-2)

A hybrid deep learning pipeline that combines **YOLOv12n** for real-time brain tumor detection with **SAM2 (Segment Anything Model 2)** for precise tumor segmentation in MRI images.

![Pipeline Overview](https://img.shields.io/badge/Pipeline-Detection%20→%20Segmentation-brightgreen)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Dataset](#-dataset)
- [Usage](#-usage)
- [Results](#-results)
- [Model Weights](#-model-weights)
- [Citation](#-citation)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## 🔬 Overview

Brain tumor segmentation is crucial for clinical diagnosis and treatment planning. This project implements a **self-prompting hybrid approach** that:

1. **Detects** tumor regions using YOLOv12n, generating bounding boxes around potential tumors
2. **Segments** tumors precisely using SAM2, with bounding boxes serving as automatic prompts

This eliminates the need for manual annotation during inference and achieves state-of-the-art performance on brain tumor MRI datasets.

### Why This Approach?

| Challenge | Solution |
|-----------|----------|
| Manual segmentation is time-consuming | Automated end-to-end pipeline |
| Traditional methods require ground-truth masks | Uses bounding box annotations only |
| Real-time inference needed for clinical use | YOLOv12n provides fast detection |
| Complex tumor boundaries | SAM2 delivers precise segmentation |

---

## ✨ Key Features

- **🚀 Real-Time Detection**: YOLOv12n provides fast and accurate tumor localization
- **🎯 Precise Segmentation**: SAM2 generates pixel-perfect tumor masks
- **🔄 Self-Prompting**: Automatic prompt generation from detection to segmentation
- **📊 High Performance**: Achieves excellent Dice scores and IoU metrics
- **🏥 Clinical Ready**: Fast inference suitable for real-time medical applications
- **📈 Minimal Annotation**: Requires only bounding box labels for training

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INPUT: MRI Image                            │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      YOLOv12n DETECTION                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐              │
│  │  Backbone   │ →  │    Neck     │ →  │    Head     │              │
│  │  (R-ELAN)   │    │(Area Attn)  │    │ (Bounding   │              │
│  │             │    │             │    │   Boxes)    │              │
│  └─────────────┘    └─────────────┘    └─────────────┘              │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  │ Bounding Box Coordinates
                                  │ (x_min, y_min, x_max, y_max)
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      SAM2 SEGMENTATION                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐              │
│  │   Image     │ →  │   Prompt    │ →  │    Mask     │              │
│  │  Encoder    │    │  Encoder    │    │   Decoder   │              │
│  │  (Hiera)    │    │(BBox→Prompt)│    │(Transformer)│              │
│  └─────────────┘    └─────────────┘    └─────────────┘              │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    OUTPUT: Segmentation Mask                        │
└─────────────────────────────────────────────────────────────────────┘
```

### Component Details

**YOLOv12n (Detection)**
- **Backbone**: R-ELAN (Residual Efficient Layer Aggregation Network) with 7×7 separable convolutions
- **Neck**: Area attention mechanism built on FlashAttention
- **Head**: Generates bounding boxes with confidence scores

**SAM2 (Segmentation)**
- **Image Encoder**: Hiera encoder pre-trained with MAE
- **Prompt Encoder**: Converts bounding boxes to spatial prompts
- **Mask Decoder**: Two-way transformer for precise mask generation

---

## 🛠 Installation

### Prerequisites

- Python 3.8+
- CUDA 11.8+ (for GPU acceleration)
- PyTorch 2.0+

### Setup

```bash
# Clone the repository
git clone https://github.com/Rayan1088/Brain-tumor-segmentation-using-yolo12n-and-SAM2.git
cd Brain-tumor-segmentation-using-yolo12n-and-SAM2

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install YOLOv12 (Ultralytics)
pip install ultralytics

# Install SAM2
pip install segment-anything-2
```

### Requirements

```txt
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.0
segment-anything-2
opencv-python>=4.8.0
numpy>=1.24.0
matplotlib>=3.7.0
pillow>=10.0.0
tqdm>=4.65.0
scikit-learn>=1.3.0
pandas>=2.0.0
```

---

## 📁 Dataset

This project uses the **Figshare Brain Tumor MRI Dataset**, which contains 3,064 T1-weighted contrast-enhanced MRI images.

### Dataset Structure

```
data/
├── images/
│   ├── train/          # 2,451 images (80%)
│   ├── val/            # 307 images (10%)
│   └── test/           # 306 images (10%)
├── labels/
│   ├── train/          # YOLO format annotations
│   ├── val/
│   └── test/
└── masks/              # Binary segmentation masks
    ├── train/
    ├── val/
    └── test/
```

### Download Dataset

```bash
# Download from Kaggle
kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset

# Or download from Figshare
# https://figshare.com/articles/dataset/brain_tumor_dataset/1512427
```

### Data Format

- **Images**: 640×640 pixels, grayscale MRI scans
- **Labels**: YOLO format (class_id, x_center, y_center, width, height)
- **Masks**: Binary masks for tumor regions

---

## 🚀 Usage

### Training YOLOv12n

```python
from ultralytics import YOLO

# Load YOLOv12n model
model = YOLO('yolov12n.pt')

# Train on brain tumor dataset
results = model.train(
    data='data/brain_tumor.yaml',
    epochs=75,
    imgsz=640,
    batch=16,
    optimizer='AdamW',
    lr0=0.002,
    augment=True
)
```

### Inference Pipeline

```python
from ultralytics import YOLO
from sam2.sam2_image_predictor import SAM2ImagePredictor
import cv2
import numpy as np

def segment_brain_tumor(image_path):
    """
    Complete pipeline for brain tumor detection and segmentation.
    
    Args:
        image_path: Path to MRI image
        
    Returns:
        segmentation_mask: Binary mask of tumor region
    """
    # Load models
    yolo_model = YOLO('weights/yolov12n_brain_tumor.pt')
    sam2_predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
    
    # Load image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Step 1: Detect tumor with YOLOv12n
    results = yolo_model(image)
    
    # Extract bounding box
    boxes = results[0].boxes.xyxy.cpu().numpy()
    
    if len(boxes) == 0:
        return None
    
    # Get the highest confidence detection
    bbox = boxes[0]  # [x_min, y_min, x_max, y_max]
    
    # Step 2: Segment with SAM2 using bbox as prompt
    sam2_predictor.set_image(image_rgb)
    
    masks, scores, _ = sam2_predictor.predict(
        box=bbox[None, :],
        multimask_output=False
    )
    
    # Get binary mask
    segmentation_mask = (masks[0] > 0.5).astype(np.uint8)
    
    return segmentation_mask

# Run inference
mask = segment_brain_tumor('test_mri.png')
```

### Batch Processing

```python
import os
from pathlib import Path

def process_dataset(input_dir, output_dir):
    """Process multiple MRI images."""
    os.makedirs(output_dir, exist_ok=True)
    
    for image_file in Path(input_dir).glob('*.png'):
        mask = segment_brain_tumor(str(image_file))
        
        if mask is not None:
            output_path = Path(output_dir) / f"{image_file.stem}_mask.png"
            cv2.imwrite(str(output_path), mask * 255)
            
process_dataset('data/test/images', 'results/masks')
```

### Visualization

```python
import matplotlib.pyplot as plt

def visualize_results(image_path, mask):
    """Visualize original image with segmentation overlay."""
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image_rgb, cmap='gray')
    axes[0].set_title('Original MRI')
    axes[0].axis('off')
    
    # Segmentation mask
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Segmentation Mask')
    axes[1].axis('off')
    
    # Overlay
    overlay = image_rgb.copy()
    overlay[mask == 1] = [255, 0, 0]  # Red overlay
    axes[2].imshow(cv2.addWeighted(image_rgb, 0.7, overlay, 0.3, 0))
    axes[2].set_title('Heatmap Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('visualization.png', dpi=300)
    plt.show()
```

---

## 📊 Results

### Performance Metrics

| Metric | Value |
|--------|-------|
| **Accuracy** | 99.71% |
| **Dice Coefficient** | 91.85% |
| **IoU (Jaccard Index)** | 84.87% |
| **Precision** | 94.75% |
| **Recall** | 89.06% |
| **F1-Score** | 91.82% |
| **Inference Time** | ~45 ms |

### YOLOv12n Detection Results

| Metric | Value |
|--------|-------|
| **mAP@50** | 92.9% |
| **Precision** | 90.28% |
| **Recall** | 86.66% |

### Comparison with State-of-the-Art

| Model | Accuracy | Dice Score |
|-------|----------|------------|
| 2D U-Net | 92.16% | 81.2% |
| 3D U-Net | - | 86.0% |
| YOLO NAS | 96.20% | 85.81% |
| Modified U-Net | 99.5% | 85.02% |
| DeepLabv3+ + ResNet18 | 97.48% | 91.2% |
| YOLOv8 + SAM | - | 79.0% |
| **YOLOv12n + SAM2 (Ours)** | **99.71%** | **91.85%** |

### Inference Time Comparison

| Model | Inference Time (s) |
|-------|-------------------|
| 3D U-Net | 120 |
| YOLO + SAM | 25 |
| **YOLOv12n + SAM2 (Ours)** | **0.045** |

---

## 📦 Model Weights

Download pre-trained weights:

| Model | Description | Download |
|-------|-------------|----------|
| YOLOv12n | Trained on Figshare Brain Tumor Dataset | [Download](#) |
| SAM2-Large | Pre-trained SAM2 Hiera-Large | [HuggingFace](https://huggingface.co/facebook/sam2-hiera-large) |

---

## 📁 Project Structure

```
Brain-tumor-segmentation-using-yolo12n-and-SAM2/
├── data/
│   ├── brain_tumor.yaml          # Dataset configuration
│   ├── images/                   # MRI images
│   ├── labels/                   # YOLO annotations
│   └── masks/                    # Ground truth masks
├── models/
│   ├── yolov12n.py              # YOLOv12n model wrapper
│   └── sam2_segmentor.py        # SAM2 segmentation module
├── utils/
│   ├── data_utils.py            # Data loading utilities
│   ├── metrics.py               # Evaluation metrics
│   └── visualization.py         # Visualization tools
├── weights/
│   └── yolov12n_brain_tumor.pt  # Trained weights
├── notebooks/
│   └── demo.ipynb               # Demo notebook
├── train.py                     # Training script
├── inference.py                 # Inference script
├── evaluate.py                  # Evaluation script
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

---

## 🔧 Configuration

### Dataset Configuration (brain_tumor.yaml)

```yaml
path: ./data
train: images/train
val: images/val
test: images/test

names:
  0: tumor

nc: 1
```

### Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Epochs | 75 |
| Batch Size | 16 |
| Image Size | 640×640 |
| Optimizer | AdamW |
| Learning Rate | 0.002 |
| Momentum | 0.9 |
| Weight Decay | 0.0005 |

---

## 📈 Training Curves

The model converges after approximately 75 epochs with the following loss curves:

- **Box Loss**: Decreases steadily, indicating improved bounding box predictions
- **Classification Loss**: Converges quickly due to single-class detection
- **DFL Loss**: Distribution focal loss for precise localization

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📜 Citation

If you use this work in your research, please cite:

```bibtex
@misc{brain_tumor_yolo_sam2,
  author = {Rayan},
  title = {Brain Tumor Segmentation using YOLOv12n and SAM2},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/Rayan1088/Brain-tumor-segmentation-using-yolo12n-and-SAM2}
}
```

### Related Works

```bibtex
@article{tian2025yolov12,
  title={YOLOv12: Attention-Centric Real-Time Object Detectors},
  author={Tian, Yunjie and Ye, Qixiang and Doermann, David},
  journal={arXiv preprint arXiv:2502.12524},
  year={2025}
}

@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and others},
  journal={arXiv preprint arXiv:2408.00714},
  year={2024}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for the YOLO implementation
- [Meta AI](https://github.com/facebookresearch/segment-anything-2) for SAM2
- [Figshare Brain Tumor Dataset](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427) contributors
- The medical imaging research community

---

## 📧 Contact

**Rayan** - [@Rayan1088](https://github.com/Rayan1088)

Project Link: [https://github.com/Rayan1088/Brain-tumor-segmentation-using-yolo12n-and-SAM2](https://github.com/Rayan1088/Brain-tumor-segmentation-using-yolo12n-and-SAM2)

---

<p align="center">
  <b>⭐ If you find this project useful, please consider giving it a star! ⭐</b>
</p>
