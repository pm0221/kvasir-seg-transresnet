# Kvasir-SEG Image Segmentation Using TransResNet

## Project Overview
This project implements a deep learning model for medical image segmentation using the Kvasir-SEG dataset. The model, named **TransResNet**, combines a ResNet18 feature-extraction branch and a Swin Transformer branch. Their features are fused, processed with an attention block, and passed through a decoder to generate a binary segmentation mask.

The implementation is provided in the Jupyter Notebook `CV_TRANSRENET.ipynb`.

## Objectives
- Download and prepare the Kvasir-SEG dataset.
- Match each endoscopic image with its corresponding segmentation mask.
- Preprocess images and masks and apply data augmentation.
- Build a hybrid ResNet18–Swin Transformer segmentation architecture.
- Train the model using a combined Binary Cross-Entropy and Dice loss.
- Save and reload the model with the best validation loss.
- Evaluate segmentation performance on the test set.

## Dataset
The notebook downloads the **Kvasir-SEG** dataset using KaggleHub:

```python
kagglehub.dataset_download("debeshjha1/kvasirseg")
```

The code searches the downloaded dataset for folders named `images` and `masks`. It supports image files with `.jpg`, `.jpeg`, and `.png` extensions, and mask files with `.png`, `.jpg`, and `.jpeg` extensions.

Image and mask files are paired by matching their filenames (without extensions). Only matching pairs are included in the dataset.

## Project Workflow

### 1. Environment and Libraries
The notebook installs and uses the following main libraries:
- PyTorch – model building and training
- timm – pretrained ResNet18 and Swin Transformer models
- Albumentations – image and mask transformations
- NumPy – array operations
- Matplotlib – visualization support
- Pillow – image loading
- scikit-learn – dataset splitting
- tqdm – training and testing progress bars
- KaggleHub – dataset download

### 2. Configuration
The notebook defines these training settings:

| Parameter | Value |
|---|---:|
| Random seed | 42 |
| Image size | 224 × 224 |
| Batch size | 4 |
| Epochs | 20 |
| Learning rate | 0.0001 |
| Optimizer | AdamW |
| Weight decay | 0.0001 |

The notebook automatically selects CUDA when a compatible GPU is available; otherwise, it uses the CPU.

### 3. Dataset Preparation
- Locate the `images` and `masks` folders.
- Collect image and mask file paths.
- Match image-mask pairs using common filenames.
- Split the matched pairs into training, validation, and testing sets:
  - Training: 80%
  - Validation: 10%
  - Testing: 10%

The split uses `random_state=42`.

### 4. Image Preprocessing and Augmentation
All images and masks are resized to 224 × 224 pixels.

Training transformations include:
- Horizontal flip with probability 0.5
- Vertical flip with probability 0.2
- Random 90-degree rotation with probability 0.5
- Shift, scale, and rotation augmentation

Images are normalized using the ImageNet mean and standard deviation, then converted to PyTorch tensors. Validation and test images are resized and normalized without the random training augmentations.

Masks are loaded in grayscale and converted to binary masks using a pixel threshold greater than 127.

### 5. TransResNet Architecture
The custom **TransResNet** model contains the following components:

**ResNet18 branch**
- Uses pretrained ResNet18 weights.
- Extracts hierarchical image features at multiple stages.
- Uses feature maps from the selected intermediate layers.

**Swin Transformer branch**
- Uses the pretrained `swin_tiny_patch4_window7_224` model.
- Extracts features at four stages.
- Converts the feature-map layout to channel-first format and resizes the maps to match the corresponding ResNet feature maps.

**Feature fusion**
- Concatenates corresponding ResNet18 and Swin Transformer feature maps.
- Uses 1 × 1 convolution layers to fuse the concatenated features.

**Attention block**
- Applies multi-head self-attention with 8 heads.
- Uses layer normalization, residual connections, and an MLP with GELU activation.

**Decoder**
- Uses decoder blocks with bilinear upsampling.
- Combines upsampled features with skip features.
- Applies convolution, batch normalization, and ReLU activation.

**Output layer**
- Produces a single-channel segmentation output at 224 × 224 resolution.
- Returns logits, which are converted to probabilities using sigmoid during metric calculation.

### 6. Loss Function
The notebook combines two loss functions:

- **Binary Cross-Entropy with Logits Loss (BCEWithLogitsLoss):** Measures pixel-wise binary classification error.
- **Dice Loss:** Measures the overlap between predicted and ground-truth masks.

The combined loss is:

```text
Combined Loss = BCE Loss + Dice Loss
```

### 7. Model Training
- Optimizer: AdamW
- Learning rate: 0.0001
- Weight decay: 0.0001
- Batch size: 4
- Maximum epochs: 20

After each epoch, the model is evaluated on the validation set. Whenever validation loss improves, the model weights are saved to `transresnet_best.pth`.

### 8. Model Evaluation
The best saved model is loaded and evaluated on the test dataset. Predicted probabilities are converted into binary masks using a threshold of 0.5.

The notebook calculates:
- **Dice Similarity Coefficient (DSC):** Measures overlap between predicted and actual masks.
- **Intersection over Union (IoU):** Measures the intersection divided by the union of predicted and actual masks.
- **F1 Score:** Harmonic mean of precision and recall.
- **Accuracy:** Proportion of correctly classified pixels.

The metric values are printed when the notebook is executed.

## Technologies and Libraries
- Python
- PyTorch
- Torchvision-compatible model ecosystem through timm
- Swin Transformer
- ResNet18
- Albumentations
- NumPy
- Matplotlib
- Pillow
- scikit-learn
- KaggleHub

## Installation

Install the required packages:

```bash
pip install torch torchvision timm albumentations numpy matplotlib pillow scikit-learn tqdm kagglehub
```

## How to Run
1. Open the notebook in Jupyter Notebook, JupyterLab, or Google Colab.
2. Install the required dependencies.
3. Run the cells in order.
4. Allow KaggleHub to download the Kvasir-SEG dataset.
5. Confirm that the downloaded dataset contains folders named `images` and `masks`.
6. Train the model and allow the best checkpoint to be saved.
7. Run the final evaluation cells to view the test metrics.

A CUDA-enabled GPU is recommended for training, but the notebook can select CPU when CUDA is unavailable.

## Project Structure

```text
Kvasir_SEG_TransResNet/
├── CV_TRANSRENET.ipynb
├── README.md
└── transresnet_best.pth   # Generated during training
```

The dataset is downloaded by the notebook and is not included in this project folder by default.

## Notes
- The notebook uses only image-mask pairs with matching filenames.
- Training augmentations are applied to both images and masks through the same Albumentations transformation pipeline.
- The model checkpoint is saved whenever validation loss improves.
- The notebook reports DSC, IoU, F1 score, and pixel accuracy; actual values depend on the training run and dataset.
- This is an image-segmentation project and is not intended to provide a clinical diagnosis.

## Author
Prayag Maru
