# MNIST Digit Classification with CNN

This project implements a Convolutional Neural Network (CNN) using PyTorch to classify handwritten digits from the MNIST dataset. The model achieved **99.04% accuracy** on the test set.

## 📌 Features
- Implements a **CNN** with two convolutional layers.
- Uses **PyTorch** for model training and evaluation.
- **Adam optimizer** for better convergence.
- **CrossEntropyLoss** for multi-class classification.
- **Normalization** for improved learning.
- **GPU support** for accelerated training.

## 🚀 Getting Started
### 1️⃣ Install Dependencies
```bash
pip install torch torchvision matplotlib numpy
```

### 2️⃣ Run the Training Script
```bash
python train.py
```

## 📊 Model Architecture
| Layer      | Type              | Output Shape |
|------------|------------------|--------------|
| Conv1      | 3x3, 32 Filters  | (32, 26, 26) |
| MaxPool    | 2x2              | (32, 13, 13) |
| Conv2      | 3x3, 16 Filters  | (16, 11, 11) |
| MaxPool    | 2x2              | (16, 5, 5)   |
| Flatten    | -                | (400)        |
| FC1        | Linear(400, 120) | (120)        |
| FC2        | Linear(120, 84)  | (84)         |
| FC3        | Linear(84, 10)   | (10)         |

## 🎯 Results
- **Final Accuracy:** 99.04%
- **Loss Function:** CrossEntropyLoss
- **Optimizer:** Adam (lr=0.0005)

## 📝 Notes
- Ensure your device supports CUDA for GPU acceleration.
- Modify `num_epochs`, `batch_size`, and `learning_rate` in the script to experiment.

## 📌 Future Improvements
- Implement **Dropout** to reduce overfitting.
- Experiment with **Batch Normalization**.
- Extend to **CIFAR-10 or custom datasets**.

---
