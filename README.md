# CNN-Deepfake-Classification
# 🏆 Image Classification with Ensemble CNNs

> **Top 15% Solution** for Kaggle Deepfake Classification Competition  
> An ensemble learning approach combining complementary CNN architectures

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Architecture](#architecture)
- [The Journey](#the-journey)
- [Dataset & Preprocessing](#dataset--preprocessing)
- [Training Strategy](#training-strategy)
- [Results Analysis](#results-analysis)
- [Lessons Learned](#lessons-learned)
- [Installation & Usage](#installation--usage)

---

## 🎯 Overview

This project tackles a challenging 5-class image classification problem using an ensemble learning strategy. Rather than relying on a single model, I developed two complementary CNN architectures that work together through soft voting to achieve superior performance.

**Why Ensemble Learning?**

Different CNN architectures capture different patterns in the data. By combining their predictions, we leverage their complementary strengths while mitigating individual weaknesses. This approach proved highly effective, pushing the solution into the **top 15%** of the competition.

---

## 🏅 Key Results

| Model | Validation Accuracy | Total Errors | Improvement |
|-------|-------------------|--------------|-------------|
| **CNN v1** (Lightweight) | 92.08% | 99 | Baseline |
| **CNN v2** (Deep) | 92.16% | 98 | +0.08% |
| **🎯 Ensemble** | **93.28%** | **84** | **+1.20%** |

The ensemble reduces errors by **15 classifications** compared to the best individual model, demonstrating the power of diversity in model predictions.

---

## 🏗️ Architecture

### Two Complementary Approaches

#### 🔹 CNN v1: Lightweight & Efficient

A streamlined architecture designed for computational efficiency while maintaining strong baseline performance.

```
Input (3×100×100)
    ↓
Conv Block 1: 3→64 channels
    ↓ MaxPool
Conv Block 2: 64→96 channels
    ↓ MaxPool
Conv Block 3: 96→128 channels
    ↓ MaxPool + Dropout(0.2)
Adaptive AvgPool (2×2)
    ↓
FC1: 512→256 + Dropout(0.4)
    ↓
FC2: 256→5 (output)
```

**Strengths:**
- Fast training and inference
- Good generalization with minimal parameters
- Excellent performance on distinct classes

#### 🔹 CNN v2: Deep & Expressive

A deeper architecture with enhanced representational capacity for capturing complex patterns.

```
Input (3×100×100)
    ↓
Conv Block 1: 3→64 channels
    ↓ MaxPool
Conv Block 2: 64→128 channels
    ↓ MaxPool
Conv Block 3: 128→256 channels
    ↓ MaxPool
Conv Block 4: 256→512 channels
    ↓ MaxPool + Dropout(0.4)
Adaptive AvgPool (1×1)
    ↓
FC1: 512→256 + Dropout(0.5)
    ↓
FC2: 256→5 (output)
```

**Strengths:**
- Higher channel dimensions capture nuanced features
- Better precision on ambiguous classes
- Complementary error patterns to CNN v1

### 🤝 Ensemble Strategy: Soft Voting

```python
# Weighted combination of probability distributions
ensemble_probs = 0.45 × CNN_v1_probs + 0.55 × CNN_v2_probs
final_prediction = argmax(ensemble_probs)
```

The near-equal weights (0.45/0.55) indicate both models contribute valuable, complementary information.

---

## 🚀 The Journey

### What Led to Success

#### 1. **Smart Data Augmentation**

Initially, I tried aggressive augmentation including color transformations, brightness adjustments, and contrast modifications. **This failed spectacularly** — the model couldn't learn meaningful patterns.

**The breakthrough:** Focus exclusively on **geometric transformations** that preserve essential image characteristics:

```python
transforms.Compose([
    transforms.RandomRotation(degrees=15, fill=0),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])
```

#### 2. **Batch Size Optimization**

Testing batch sizes of 16, 32, and 64 revealed dramatic differences:

- **16 & 64:** Training instability, plateaus at 90-91% accuracy
- **32:** Sweet spot providing stable gradients and consistent convergence

#### 3. **Architecture-Specific Regularization**

Different model complexities require different regularization strategies:

| Component | CNN v1 | CNN v2 | Rationale |
|-----------|--------|--------|-----------|
| Feature Dropout | 0.2 | 0.4 | Deeper model needs stronger regularization |
| FC Dropout | 0.4 | 0.5 | Prevents overfitting in high-capacity layers |
| Weight Decay | 1e-4 | 1e-4 | Consistent L2 regularization |
| Label Smoothing | 0.1 | 0.1 | Prevents overconfident predictions |

#### 4. **Learning Rate Sweet Spot**

After systematic testing:

- **0.001:** Too fast, unstable training, overshooting
- **0.0001:** Too slow, insufficient progress
- **0.0005:** ✨ Perfect balance — stable convergence without instability

#### 5. **Early Stopping Calibration**

Initial patience of 8 epochs was too aggressive. Models showed learning potential beyond temporary plateaus.

**Solution:** Increased patience to **15 epochs**, allowing models to overcome 10-15 epoch plateaus while preventing true overfitting. Final models maintained a healthy 4-5% train-validation gap.

#### 6. **Memory Management**

Training two sequential models caused GPU memory crashes. Solution:

```python
import gc

# Between model trainings
del optimizer, scheduler
gc.collect()
torch.cuda.empty_cache()
```

---

## 📊 Dataset & Preprocessing

### Dataset Structure

```
dataset/
├── train/          # Training images (PNG format)
├── validation/     # Validation images
└── test/          # Test images
```

5 classes with balanced distribution (250 samples per class in validation).

### Preprocessing Pipeline

**Training:** Geometric augmentation + Normalization (ImageNet statistics)  
**Validation/Test:** Normalization only (no augmentation)

---

## 🎓 Training Strategy

### Hyperparameters

```python
config = {
    'learning_rate': 0.0005,
    'batch_size': 32,
    'max_epochs': 150,
    'early_stopping_patience': 15,
    'optimizer': 'Adam',
    'weight_decay': 1e-4,
    'label_smoothing': 0.1
}
```

### Learning Rate Scheduling

**ReduceLROnPlateau** monitoring validation accuracy:
- Patience: 5 epochs
- Reduction factor: 0.5
- Mode: 'max' (maximize validation accuracy)

### Loss Function

**CrossEntropyLoss with Label Smoothing (ε=0.1)**
- Prevents overconfident predictions
- Improves model calibration
- Enhances generalization

---

## 📈 Results Analysis

### Per-Class Performance

#### Ensemble Model Breakdown

| Class | Precision | Recall | F1-Score | Support | Errors Reduced |
|-------|-----------|--------|----------|---------|----------------|
| **Class 0** | 0.950 | 0.952 | 0.951 | 250 | -6 |
| **Class 1** | 0.881 | 0.944 | 0.911 | 250 | -5 |
| **Class 2** | 0.960 | 0.960 | 0.960 | 250 | -1 |
| **Class 3** | 1.000 | 1.000 | 1.000 | 250 | 0 (perfect) |
| **Class 4** | 0.878 | 0.808 | 0.842 | 250 | -3 |

### Key Insights

🎯 **Class 3** achieves perfect classification across all models — indicating highly distinct visual characteristics

⚠️ **Class 4** remains most challenging — consistent confusion with Classes 0 and 1 suggests visual similarity

✨ **Ensemble Magic** — Reduces errors across ALL classes by leveraging complementary model strengths

---

## 💡 Lessons Learned

### ✅ What Worked

1. **Ensemble Learning** — 1.2% accuracy boost from diversity
2. **Geometric-Only Augmentation** — Preserves essential features
3. **Architecture Diversity** — Lightweight + Deep models complement each other
4. **Patience in Training** — Allowing models to overcome plateaus
5. **Batch Size 32** — Optimal stability-convergence balance

### ❌ What Didn't Work

1. **Aggressive Color Augmentation** — Introduced too much noise
2. **Extreme Batch Sizes** (16, 64) — Training instability
3. **Low Patience** (8 epochs) — Premature stopping
4. **Extreme Learning Rates** — Either unstable or too slow

### 🔑 Critical Success Factors

- **Differentiated Regularization:** Tailor dropout to architecture complexity
- **Memory Management:** Explicit cleanup between training phases
- **Systematic Hyperparameter Search:** Test methodically, not randomly
- **Equal Weight Ensemble:** Both models contribute valuable information

---

## 🛠️ Installation & Usage

### Requirements

```bash
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn
```

### Quick Start

```python
# Load trained models
model_v1 = CNNv1().to(device)
model_v2 = CNNv2().to(device)

model_v1.load_state_dict(torch.load('best_cnn_v1.pth'))
model_v2.load_state_dict(torch.load('best_cnn_v2.pth'))

# Ensemble prediction
def ensemble_predict(img_tensor, weights=[0.45, 0.55]):
    with torch.no_grad():
        probs_v1 = F.softmax(model_v1(img_tensor), dim=1)
        probs_v2 = F.softmax(model_v2(img_tensor), dim=1)
        
        ensemble_probs = weights[0] * probs_v1 + weights[1] * probs_v2
        prediction = torch.argmax(ensemble_probs, dim=1)
    
    return prediction
```

### Training from Scratch

```bash
python train.py --config config.yaml
```

---

## 🎖️ Competition Achievement

**Top 15%** finish in Kaggle Deepfake Classification Competition

This result demonstrates that careful architecture design, systematic hyperparameter tuning, and ensemble methods can compete with much larger, more complex models.

---

## 📝 Citation

```bibtex
@misc{gheorghe2025imageclassification,
  author = {Gheorghe Bogdan-Alexandru},
  title = {Image Classification using CNNs with Ensemble Learning},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yourusername/project}}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- University of Bucharest, Faculty of Mathematics and Computer Science
- Kaggle community for the challenging competition
- PyTorch team for the excellent framework

---

**Built with ❤️ and lots of hyperparameter tuning**
