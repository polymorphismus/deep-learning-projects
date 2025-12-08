# Deep Learning Projects

This repo contains eight deep learning projects across different domains, including computer vision and natural language processing.

## Overview

This collection includes projects covering:
- **Transfer Learning**: Fine-tuning pre-trained models for image classification
- **Convolutional Neural Networks**: Building and training CNNs from scratch for image classification
- **Super-Resolution**: Enhancing image quality using deep learning techniques
- **Natural Language Processing**: Word embeddings, text classification, and named entity recognition
- **Biomedical Image Analysis**: Cell counting using density maps and CNNs
- **Question Answering**: Transformer fine-tuning for extractive QA
- **Retrieval-Augmented Generation**: Vector search + LLM answering

---

## Projects

### 1. Transfer Learning for Image Classification
**Notebook**: [transfer_learning_image_classification.ipynb](./transfer_learning_image_classification_fin.ipynb)

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
**Notebook**: [cifar_10_fcn_cnn.ipynb](./cifar_10_fin.ipynb)

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
**Notebook**: [Super_Resolution.ipynb](./Super_Resolution.ipynb)

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

### 4. Word Embedding & Genre Classification
**Notebook**: [word_embeddings_and_classification.ipynb](./word_embedding_fin.ipynb)

**Summary**: This project builds a complete NLP pipeline for analyzing and classifying song lyrics using custom-trained word embeddings. It combines classical machine learning, modern deep learning, and multiple semantic evaluation techniques to explore how meaning is captured in text representations. The workflow starts from raw MetroLyrics data and progresses through preprocessing, word embedding training, sentiment modeling, semantic visualization, and multi-method genre classification, including a convolutional neural network.

**Key Techniques**:
- **Text Preprocessing**:
  - Lowercasing, punctuation removal, tokenization
  - Stopword removal and lemmatization
  - Rare word filtering (frequency ≤ 5) to reduce noise
  - Per-genre word frequency analysis

- **Word2Vec Embeddings**:
  - Skip-gram model implementation using Gensim
  - Training with multiple window sizes (2, 10, 20) to capture local vs. global context
  - 300-dimensional embeddings with vocabulary of 36,418 words
  - Word similarity analysis and vector algebra (e.g., king - man + woman ≈ queen)

- **Sentiment Analysis**:
  - Sentiment regression using Ridge and MLP models
  - Mapping word embeddings to sentiment scores from SemEval-2015 lexicon
  - Predicting sentiment for words not in the original lexicon
  - Linear Ridge model outperformed MLP, indicating linear sentiment structure in embeddings

- **Genre Classification**:
  - Bag-of-Words (BOW) with bigrams + Multinomial Naive Bayes baseline
  - Average Word2Vec embeddings + Logistic Regression
  - TF-IDF weighted embeddings for genre classification
  - Comparison of semantic vs. frequency-based representations

**Dataset**:
- **Source**: MetroLyrics dataset
- **Size**: 49,976 songs with full lyrics
- **Genres**: Multiple genres including Rock, Pop, and others
- **Features**: Song title, artist, year, genre, lyrics, sentiment labels

**Results**:
- **Word2Vec Training**: Successfully trained embeddings capturing semantic relationships
- **Sentiment Regression**: Ridge model achieved CV MSE of 0.076, demonstrating linear sentiment structure
- **Genre Classification**: BOW + Naive Bayes achieved 61.2% accuracy as baseline
- **Key Insight**: Word embeddings capture meaningful semantic and emotional structure, enabling downstream tasks like sentiment prediction and genre classification

---

### 5. Cell Counting with CNNs and Density Maps
**Notebook**: [counting_cell_cnn_density.ipynb](./cell_count.ipynb)

**Summary**: This project implements and compares several approaches for cell counting in microscopy images, using both regression-based CNNs and fully convolutional architectures that learn density maps. The goal is to estimate the number of fluorescent cell nuclei in synthetic images containing hundreds of small circular objects. The project progresses through three modeling families: baseline CNN regressor, feature-rich CNN regressor, and UNet-based density estimation model.

**Key Techniques**:
- **Baseline CNN Regressor**:
  - Minimal convolutional model treating cell counting as pure regression
  - Convolutional blocks extracting global features
  - Global pooling followed by fully connected layer outputting single scalar count
  - Direct mapping from image appearance to total cell count

- **Feature-Rich CNN Regressor (CellCountCNNRegressor)**:
  - Deeper CNN with more convolutional blocks
  - Larger receptive field to capture crowded regions
  - Improved ability to handle dense cell distributions

- **UNet-based Density Estimation**:
  - Fully convolutional network outputting pixel-wise density maps
  - Density map integral approximates total cell count
  - Standard approach in modern crowd-counting and microscopy papers
  - Learns spatial structure rather than only scalar target

- **Data Processing**:
  - Custom dataset class for paired images and binary label masks
  - Conversion of masks to ground-truth density maps using Gaussian filtering
  - Data augmentation (horizontal/vertical flips, rotations)
  - ImageNet normalization for transfer learning compatibility

**Dataset**:
- **Task**: Cell counting in synthetic microscopy images
- **Training Images**: 180 images
- **Validation Images**: 20 images
- **Format**: Paired RGB images and binary label masks
- **Challenge**: Counting hundreds of small circular objects (fluorescent cell nuclei) per image

**Architecture**:
- **Baseline CNN**: 3 convolutional blocks (32 → 64 → 128 channels) with max pooling, followed by fully connected layers
- **Feature-Rich CNN**: Deeper architecture with expanded receptive fields
- **UNet**: Encoder-decoder architecture with skip connections for density map prediction

**Results**:
- **Baseline CNN**: Best validation MAE of 32.67 after 20 epochs
- **Training Progress**: Improved from initial MAE of 60.41 to final validation MAE of 32.67
- **Key Insight**: Density map approaches (UNet) provide spatial information and are standard in modern cell counting, while regression CNNs offer simpler direct count prediction

**Applications**: Biomedical image analysis, cell counting in microscopy images, crowd counting, object density estimation

---

### 6. Named Entity Recognition with Cross-Dataset Label Mapping
**Notebook**: [ner_project_fin.ipynb](./ner_project_fin.ipynb)

**Summary**: This project investigates the critical issue of label space mismatches when applying Named Entity Recognition (NER) models trained on one dataset to another. The project examines how mismatched label spaces distort predictions, how subword tokenization complicates label alignment, and why naïve fine-tuning of pretrained NER models may fail when parts of the classifier head receive no supervision. The goal is to demonstrate that resolving label-space alignment is essential before any meaningful fine-tuning can occur.

**Key Objectives**:
- Investigate label space mismatches between pretrained models and target datasets
- Understand how subword tokenization affects label alignment in NER tasks
- Demonstrate the importance of proper label-space alignment for successful fine-tuning
- Compare fine-tuning strategies with mismatched vs. properly aligned label spaces

**Key Techniques**:
- **Label Alignment with Subword Tokenization**:
  - Alignment function to map word-level NER labels to subword tokens
  - Strategy: Assign entity label only to first subtoken of each word
  - Assign -100 to inner subtokens and special tokens ([CLS], [SEP], [PAD]) to ignore during loss computation
  - Ensures loss is computed only on first subtokens

- **Transfer Learning for NER**:
  - Using pretrained BERT models (dslim/bert-base-NER, bert-base-cased)
  - Fine-tuning on WikiANN dataset with proper label space initialization
  - Comparison of mismatched vs. properly aligned classifier heads

- **Evaluation**:
  - Using seqeval metric for sequence labeling evaluation
  - Computing precision, recall, F1-score, and accuracy
  - Handling label space mismatches (e.g., collapsing MISC labels)

**Dataset**:
- **Source**: WikiANN dataset (English)
- **Task**: Named Entity Recognition
- **Training Samples**: 20,000 sentences
- **Validation Samples**: 10,000 sentences
- **Test Samples**: 10,000 sentences
- **Label Schema**: 7 labels (O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC)
- **Challenge**: Pretrained model (dslim/bert-base-NER) was trained on CoNLL-2003 with 4 entity types (PER, ORG, LOC, MISC), while WikiANN uses only 3 (PER, ORG, LOC) without MISC

**Architecture**:
- **Pretrained Model**: dslim/bert-base-NER (BERT-base trained on CoNLL-2003)
- **Base Model**: bert-base-cased with properly initialized classifier head
- **Tokenization**: BERT tokenizer with subword tokenization
- **Classification Head**: Token classification head matching dataset label space

**Results**:
- **Mismatched Model (dslim/bert-base-NER)**: F1-score of **0.04** on WikiANN, demonstrating severe performance degradation due to label space mismatch
- **Fine-tuned Mismatched Model**: Identical performance (F1 ≈ 0.04) after 3 epochs, showing that fine-tuning cannot succeed with mismatched label spaces
- **Properly Aligned Model (bert-base-cased)**: After 5 epochs of fine-tuning with correct label space:
  - **Precision**: 0.83
  - **Recall**: 0.85
  - **F1-score**: 0.84
  - **Accuracy**: 0.93
- **Key Insight**: Fine-tuning cannot succeed when the model's label space does not match the dataset. Once label mismatch was resolved, BERT learned WikiANN effectively and achieved high-quality NER performance

**Applications**: Named entity recognition, information extraction, cross-dataset model adaptation, transfer learning for sequence labeling tasks

---

### 7. Question Answering with Transformers
**Notebook**: [qa_squad_fin.ipynb](./qa_squad_fin.ipynb)

**Summary**: Fine-tunes transformer models on SQuAD v1.1, covering preprocessing, span alignment, training, and evaluation with EM/F1. Compares performance across context and answer lengths to understand model sensitivity.

**Key Techniques**:
- Tokenization and sequence length management for extractive QA
- Fine-tuning BERT-based models on SQuAD v1.1
- Evaluation with Exact Match and F1 metrics
- Performance analysis by context length and answer length

**Dataset**:
- **SQuAD v1.1**: 100k+ question-answer pairs over Wikipedia articles

---

### 8. Retrieval-Augmented Generation on “Winnie-the-Pooh”
**Notebook**: [pooh_rag.ipynb](./pooh_rag.ipynb)

**Summary**: Builds a RAG pipeline to answer questions about A.A. Milne’s “Winnie-the-Pooh,” combining document chunking, embedding, FAISS vector search, and LLM-based answer generation. Includes prompt tuning and retrieval experiments to balance precision and recall.

**Key Techniques**:
- Sentence-transformer embeddings with document chunking
- FAISS vector store for similarity search
- LangChain-style pipeline for retrieval + generation
- Prompt engineering to improve grounded answers
- Evaluation of chunking/embedding variants on answer quality
