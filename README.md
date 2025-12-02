# Deep Learning Projects

This repo contains three comprehensive deep learning projects focusing on computer vision and image processing.


## Overview

This collection includes projects covering:
- **Transfer Learning**: Fine-tuning pre-trained models for image classification
- **Convolutional Neural Networks**: Building and training CNNs from scratch for image classification
- **Super-Resolution**: Enhancing image quality using deep learning techniques

---

## Projects

### 1. Transfer Learning for Image Classification
**File**: `transfer_learning_image_classification_fin.ipynb`

**Summary**: This project explores transfer learning for a 9-class image classification task. The project evaluates multiple pretrained architectures, experiments with augmentation pipelines, analyzes model errors, and introduces targeted regularization strategies to reduce overfitting and improve generalization.

**Key Objectives**:
- Implement and compare multiple pre-trained CNN architectures for image classification
- Evaluate the effectiveness of transfer learning on a small, domain-specific dataset
- Conduct comprehensive error analysis to identify model weaknesses
- Experiment with data augmentation strategies to improve generalization
- Optimize model performance through hyperparameter tuning and architecture selection

**Key Techniques**:
- Transfer learning with VGG16, ResNet18, DenseNet161, GoogleNet, MobileNet-v2, ResNeXt50, and Wide ResNet-50-2 architectures
- Fine-tuning: Initializing networks with pre-trained weights and training all layers with a lower learning rate
- Architecture adaptation: Replacing final classification layers to match target task (9 classes instead of 1000)
- Data augmentation (horizontal/vertical flips, rotations, affine transformations, color jittering)
- Model convergence visualization and analysis
- Error analysis with misclassification patterns
- Class-weighted loss functions for imbalanced datasets
- Learning rate scheduling and optimization

**Dataset**:
- **Task**: Multi-class image classification
- **Classes**: 9 Israeli politicians
- **Training Images**: ~929 images (~100 per class)
- **Validation Images**: ~234 images
- **Challenge**: Small dataset size requires transfer learning for effective generalization

**Results**: Best validation accuracy of **90%** using Wide ResNet-50-2 architecture with optimized data augmentation and training strategies.

---

### 2. CIFAR-10 Classification with Fully Connected and Convolutional Networks
**File**: `cifar_10_fin.ipynb`

**Summary**: This project explores image classification on the CIFAR-10 dataset using two fundamentally different model families: Fully Connected Networks (FCNs) as a baseline approach treating images as flat vectors, and Convolutional Neural Networks (CNNs) that exploit spatial structure in images. The goal is to demonstrate the progression from a simple, limited baseline toward a modern, spatially-aware model, analyzing how architectural choices, regularization, optimization, and data augmentation influence performance.

**Key Techniques**:
- **Fully Connected Networks (FCNs)**:
  - Custom neural network architecture with fully connected layers
  - Image preprocessing and normalization
  - Training loop implementation with SGD optimizer
  - Architecture variants: baseline (3072 → 100 → 20 → 10), larger hidden layers, dropout regularization, deeper networks
  - Best FCN configuration: 3072 → 512 → 256 → 10 with dropout

- **Convolutional Neural Networks (CNNs)**:
  - Custom CNN architecture with convolutional and pooling layers
  - Batch normalization for training stability
  - Dropout regularization to prevent overfitting
  - Data augmentation (random crops, horizontal flips)
  - Multi-layer feature extraction with increasing channel depth
  - Weight decay and learning rate scheduling (CosineAnnealingLR)

**Dataset**:
- **CIFAR-10**: 60,000 color images of size 32×32, divided into 10 object categories
- **Classes**: airplane, car, bird, cat, deer, dog, frog, horse, ship, truck
- **Training Samples**: 50,000
- **Test Samples**: 10,000

**Architecture**:
- **FCN Baseline**: 3-layer fully connected network (3072 → 100 → 20 → 10 neurons) with ReLU activations
- **Best FCN**: 3072 → 512 → 256 → 10 with dropout and optimized hyperparameters
- **CNN**: 3 convolutional blocks with max pooling, followed by fully connected layers with batch normalization and dropout

**Results**:
- **FCN Performance**: Best FCN achieved ~53–55% validation accuracy, demonstrating architectural limitations
- **CNN Performance**: Custom CNN reaches ~82–83% validation accuracy with proper regularization and data augmentation
- **Key Insight**: CNNs extract hierarchical features (edges → textures → shapes → objects), which FCNs cannot, despite extensive hyperparameter tuning

---

### 3. Image Super-Resolution
**File**: `Super_Resolution.ipynb`

**Summary**: This project applies deep learning to upscale low-resolution images into higher-resolution outputs. Using the Oxford-IIIT Pet dataset, the project trains a lightweight convolutional model to reconstruct high-resolution (HR) images from bicubic-downsampled low-resolution (LR) inputs. Training combines pixel loss with VGG16-based perceptual loss, enabling the model to recover sharper edges and textures beyond standard interpolation.

**Key Components**:
- Custom dataset for LR/HR image pairs
- Bicubic downsampling and resizing
- Lightweight CNN for image-to-image regression
- Perceptual loss using VGG16 intermediate features
- Training loop with MSE + perceptual objective
- Visual comparison of LR / HR / Generated results

**Key Techniques**:
- Custom dataset class (`PetsSuperResDataset`) for low/high-resolution image pairs
- Convolutional architecture for image-to-image translation
- Perceptual loss using VGG16 feature extraction
- Bicubic interpolation for downsampling/upsampling
- Data augmentation preserving pixel neighborhood relationships (horizontal flips, color jittering)
- Lightweight 3-layer CNN architecture with large receptive fields

**Dataset**:
- **Source**: Oxford-IIIT Pet dataset
- **High-Resolution Size**: 256×256 pixels
- **Low-Resolution Size**: 128×128 pixels (downsampled, then upsampled to 256×256)
- **Split**: Train/validation split using dataset's trainval/test splits

**Architecture**:
- **Model Design**: Intentionally lightweight 3-layer convolutional network
- **Receptive Field**: Large 9×9 kernel in first layer to capture global context
- **Feature Processing**: Nonlinear feature transformation through ReLU activations
- **Output**: RGB image with final 5×5 convolution
- **Loss Function**: Combination of MSE pixel loss and VGG16-based perceptual loss

**Methodology**: The model learns to predict missing detail directly from low-resolution input, going beyond simple interpolation by learning to reconstruct high-frequency details and textures that are lost during downsampling.

---

## Technical Skills Demonstrated

### Computer Vision
- Transfer learning and fine-tuning pre-trained models
- Convolutional Neural Networks (CNNs) from scratch
- Fully Connected Networks (FCNs) for baseline comparison
- Image preprocessing and augmentation
- Super-resolution techniques
- Perceptual loss functions

### Deep Learning Techniques
- Model architecture design and optimization
- Hyperparameter tuning (learning rates, batch sizes, hidden layer sizes)
- Regularization (dropout, weight decay, batch normalization)
- Learning rate scheduling (step decay, cosine annealing)
- Error analysis and model evaluation
- Training loop implementation

### Deep Learning Frameworks
- PyTorch
- Torchvision
- Custom dataset classes
- Model training and evaluation pipelines

---

## Project Organization

Each project notebook includes:
- Complete implementation code
- Data loading and preprocessing pipelines
- Model architecture definitions
- Training loops with validation
- Results analysis and visualization
- Comprehensive documentation and insights

---

## Getting Started

To run these projects, ensure you have:
- Python 3.8+
- PyTorch and torchvision installed
- CUDA-capable GPU (recommended for faster training)
- Sufficient disk space for datasets

### Running Projects

Each notebook is self-contained and can be run independently:
1. Open the desired notebook in Jupyter or Google Colab
2. Follow the cells sequentially
3. Datasets will be downloaded automatically when needed

---
