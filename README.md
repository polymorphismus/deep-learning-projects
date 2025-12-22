# Deep Learning Projects

This repo contains applied deep learning projects across computer vision, natural language processing, and modern generative systems. The projects focus on end-to-end modeling workflows, including data preparation, model design, training, evaluation, and error analysis, with an emphasis on understanding model behavior and practical tradeoffs.

## Overview

This collection includes projects covering:
- **Biomedical Image Analysis**: Cell counting using CNNs and density maps
- **Natural Language Processing**: Named entity recognition, word embeddings, and text classification
- **Transformers**: Fine-tuning for sequence labeling and question answering
- **Retrieval-Augmented Generation**: Vector search combined with LLM-based answering
- **Transfer Learning**: Fine-tuning pretrained models under small-data constraints
- **Convolutional Neural Networks**: Architectural comparisons and regularization
- **Image-to-Image Learning**: Super-resolution with perceptual loss

---

## Projects

### 1. Cell Counting with CNNs and Density Maps
**Notebook**: [counting_cell_cnn_density.ipynb](./cell_count.ipynb)

**Summary**:  
This project addresses the problem of counting objects in microscopy images, a common task in biomedical image analysis. Multiple modeling approaches are implemented and compared, progressing from direct regression to fully convolutional density-map estimation.

The project begins with a baseline CNN regressor that predicts a scalar count directly from the image, followed by a deeper feature-rich regressor with an expanded receptive field. It then introduces a UNet-based fully convolutional model that predicts pixel-wise density maps whose integral corresponds to the total count. This progression highlights the limitations of pure regression approaches and motivates spatially aware models commonly used in modern counting literature.

The work includes custom dataset construction, conversion of binary masks to density maps, data augmentation, and evaluation using mean absolute error, with emphasis on architectural reasoning and error analysis.

---

### 2. Named Entity Recognition with Cross-Dataset Label Mapping
**Notebook**: [ner_project_fin.ipynb](./ner_project_fin.ipynb)

**Summary**:  
This project investigates a critical issue in applied NLP: label space mismatches when transferring pretrained Named Entity Recognition models across datasets. It demonstrates how fine-tuning can fail entirely when the classifier head does not align with the target dataset’s label schema.

The project analyzes the interaction between subword tokenization, label alignment, and loss computation, and implements a correct label-mapping strategy that enables effective fine-tuning. Models with mismatched and properly aligned label spaces are compared using standard sequence-labeling metrics.

The results show that resolving label-space mismatches is a prerequisite for successful transfer learning, emphasizing the importance of understanding model failure modes rather than focusing solely on training procedures.

---

### 3. Retrieval-Augmented Generation on “Winnie-the-Pooh”
**Notebook**: [pooh_rag.ipynb](./pooh_rag.ipynb)

**Summary**:  
This project builds a retrieval-augmented generation (RAG) pipeline to answer questions about A.A. Milne’s *Winnie-the-Pooh*. The system combines document chunking, dense embeddings, FAISS-based vector search, and LLM-based answer generation.

The project explores different chunking strategies, embedding choices, and prompt formulations, analyzing how retrieval quality impacts answer correctness and specificity. The focus is on building a clean, interpretable RAG pipeline with grounded answers rather than treating the system as a black box.

---

### 4. Transfer Learning for Image Classification
**Notebook**: [transfer_learning_image_classification.ipynb](./transfer_learning_image_classification_fin.ipynb)

**Summary**:  
This project explores transfer learning for a 9-class image classification task under small-data constraints. Multiple pretrained CNN architectures are evaluated, with experiments on augmentation pipelines, regularization strategies, and fine-tuning configurations.

The project includes detailed error analysis to identify misclassification patterns and overfitting behavior, as well as targeted adjustments to improve generalization. It highlights practical considerations when applying deep learning models to limited, domain-specific datasets.

---

### 5. Image Super-Resolution
**Notebook**: [Super_Resolution.ipynb](./Super_Resolution.ipynb)

**Summary**:  
This project applies deep learning to single-image super-resolution, reconstructing high-resolution images from low-resolution inputs. A lightweight convolutional model is trained using a combination of pixel-level loss and perceptual loss derived from pretrained VGG16 features.

The work emphasizes architectural simplicity, receptive field design, and loss-function selection, and includes visual comparisons between interpolation baselines and learned reconstructions.

---

### 6. CIFAR-10 Classification with Fully Connected and Convolutional Networks
**Notebook**: [cifar_10_fcn_cnn.ipynb](./cifar_10_fcn_cnn.ipynb)

**Summary**:  
This project compares fully connected networks and convolutional neural networks on the CIFAR-10 dataset to illustrate the impact of architectural inductive bias. FCNs are used as a baseline treating images as flat vectors, followed by CNNs that exploit spatial structure.

The experiments analyze how architectural choices, regularization, optimization, and data augmentation affect performance, demonstrating why CNNs are fundamentally better suited for image data.

---

### 7. Word Embeddings and Genre Classification
**Notebook**: [word_embeddings_and_classification.ipynb](./word_embeddings_and_classification.ipynb)

**Summary**:  
This project builds an end-to-end NLP pipeline for analyzing and classifying song lyrics using custom-trained word embeddings. It combines Word2Vec training, sentiment regression, semantic analysis, and multiple genre classification approaches.

The project compares frequency-based and embedding-based representations, highlighting how semantic structure captured by embeddings enables downstream tasks such as sentiment prediction and genre classification.

---

### 8. Question Answering with Transformers
**Notebook**: [qa_squad_fin.ipynb](./qa_squad_fin.ipynb)

**Summary**:  
This project fine-tunes transformer models on the SQuAD v1.1 dataset for extractive question answering. It covers preprocessing, answer span alignment, training, and evaluation using Exact Match and F1 metrics.

Additional analysis explores how context length and answer length affect model performance, providing insight into model sensitivity and limitations.

---

