# EEG Classification using EEGNet and TSCeption

## Introduction
This repository implements two deep learning models, EEGNet and TSCeption, for the classification of cognitive states using EEG data. EEG (Electroencephalography) is a non-invasive technique that measures electrical activity in the brain, making it valuable for understanding brain function during different cognitive tasks. The goal is to classify EEG data into distinct cognitive states based on advanced deep learning techniques.

## Methods and Techniques

### EEGNet Model
- **Model Description**: EEGNet is a deep learning architecture tailored for EEG data, designed to capture temporal and spatial dependencies effectively.
- **Key Techniques**:
  - **Temporal Convolution**: Utilizes 1D convolutions to extract temporal features from EEG signals.
  - **Depthwise and Pointwise Convolution**: Enhances model efficiency and performance.
  - **Batch Normalization**: Stabilizes and accelerates the training process.
  - **Dropout**: Mitigates overfitting by randomly dropping neurons during training.

### TSCeption Model
- **Model Description**: TSCeption integrates temporal convolutions with LSTM (Long Short-Term Memory) layers, optimizing the learning of temporal dependencies in EEG data.
- **Key Techniques**:
  - **Convolutional Layers**: Extracts hierarchical features from EEG signals.
  - **LSTM (Long Short-Term Memory)**: Captures temporal dependencies in sequential data, preserving information over time.
  - **Batch Normalization**: Normalizes activations, speeding up convergence and improving generalization.
  - **Dropout**: Regularizes the network, reducing the risk of overfitting.
  - **Sparse Categorical Crossentropy Loss**: Suitable for multi-class classification tasks, optimizing model training towards categorical targets.

## Usage
- **Dataset**: The EEG data used in this repository is sourced from the Mental Arithmetic Tasks Dataset available at PhysioNet.
- **Requirements**: Ensure you have the necessary Python libraries installed (`mne`, `tensorflow`, `scikit-learn`) as listed in `requirements.txt`.
- **Training**: Use the provided Jupyter notebooks (`EEGNet_Model.ipynb`, `TSCeption_Model.ipynb`) to train and evaluate the models on your EEG dataset.
- **Evaluation**: Assess model performance using metrics such as accuracy, precision, recall, and F1-score reported during training and validation phases.
- **Results**: Compare and analyze the performance of EEGNet and TSCeption models in classifying cognitive states based on the provided dataset.

## Conclusion
This repository serves as a comprehensive resource for implementing and evaluating EEG classification models using deep learning techniques. By leveraging EEGNet and TSCeption architectures, researchers and practitioners can advance their understanding of cognitive states through EEG data analysis.

