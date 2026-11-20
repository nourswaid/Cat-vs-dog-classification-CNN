# Cat vs Dog Image Classifier (CNN)

A convolutional neural network built with TensorFlow/Keras that classifies images as either a **cat** or a **dog**. It's a binary image classification project covering the full workflow: data preprocessing with augmentation, model building, training, evaluation, and making single predictions.

## Results

Trained for 25 epochs on 8,000 training images and validated on 2,000 test images (64x64 RGB):

| Metric | Training | Validation |
|--------|----------|------------|
| Accuracy | ~90.7% | ~83.3% |
| Loss | 0.231 | 0.445 |

The gap between training and validation accuracy suggests some overfitting after about epoch 12-14, where validation loss stops improving. See [Possible Improvements](#possible-improvements).

## Model Architecture

```
Input (64 x 64 x 3)
  -> Conv2D (32 filters, 3x3, ReLU)
  -> MaxPooling2D (2x2, stride 2)
  -> Conv2D (32 filters, 3x3, ReLU)
  -> MaxPooling2D (2x2, stride 2)
  -> Flatten
  -> Dense (128, ReLU)
  -> Dense (1, Sigmoid)
```

- **Loss:** binary cross-entropy
- **Optimizer:** Adam
- **Output:** a single sigmoid unit (probability that the image is a dog)

## Data Preprocessing

- **Training set:** pixel values rescaled to [0, 1], plus augmentation (shear 0.2, zoom 0.2, horizontal flip) to reduce overfitting.
- **Test set:** rescaling only, no augmentation.
- All images resized to 64x64, batch size 32.

## Dataset

The notebook expects this folder structure (one subfolder per class):

```
dataset/
├── training_set/
│   ├── cats/
│   └── dogs/
└── test_set/
    ├── cats/
    └── dogs/
```

The dataset is **not included** in this repo. This is the common 10,000-image cats/dogs dataset (8,000 train / 2,000 test). Download it and update the paths in the notebook to point to your local copy.

## Setup

```bash
pip install tensorflow numpy pandas matplotlib scikit-learn scipy pillow
```

Tested with Python 3.11. Note: TensorFlow >= 2.11 does not use the GPU on native Windows; use WSL2 for GPU training.

## Usage

1. Clone the repo and open `CNN.ipynb` in Jupyter.
2. Update the dataset paths in the preprocessing cells.
3. Run all cells to train the model.
4. For a single prediction, set the path to your image in Part 4:

```python
test_image = image.load_img("path/to/your/image.jpg", target_size=(64, 64))
```

The class mapping can be checked with `training_set.class_indices` (cats = 0, dogs = 1).

## Possible Improvements

- Add **Dropout** and/or early stopping to reduce overfitting
- Add more convolutional layers or use **BatchNormalization**
- Use a larger input size than 64x64
- Try **transfer learning** (e.g. MobileNetV2, ResNet50), which would likely push accuracy well above 90%
- Save the trained model (`cnn.save("model.keras")`) so it doesn't need retraining

## Tech Stack

Python, TensorFlow/Keras, NumPy, Matplotlib
