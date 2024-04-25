# Binocular ML

## Overview

This project develops a deep learning model using TensorFlow and Keras to differentiate between images of people with glasses and without glasses.

## Requirements

- Python 3.8 or higher
- Poetry for dependency management

## Setup Instructions

### 1. Install dependencies

Make sure Poetry is installed on your system. If not, install it first (official guide). Then, install all required dependencies by running:

```bash
poetry install
```

### 2. Download the dataset

Ensure your dataset is structured properly under the datasets/ directory with the subdirectories with_glasses/ and without_glasses/.

### 3. Run the Model Training

To start training the model, use the following command:

```bash
poetry run python classification_train.py
```

## Testing

If you want to test the model using your webcam, run the following command:

```bash
python webcam_test.py
```
