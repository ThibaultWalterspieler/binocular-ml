# Binocular ML

## Overview

This project develops a deep learning model using TensorFlow and Keras to differentiate between images of people with glasses and without glasses.

## Requirements

- Python 3.12
- Poetry for dependency management

## Setup Instructions

### 1. Install dependencies

Make sure Poetry is installed on your system. If not, install it first (official guide). Then, install all required dependencies by running:

```bash
poetry install
```

### 2. Download the dataset

Ensure your dataset is structured properly under the `data` directory with the subdirectories `with_glasses/` and `without_glasses/`.

### 3. Run the Model Training

To start training the model, use the following command:

```bash
poetry run python src/train.py
```

## Testing

### 1. Test with your webcam

If you want to test the model using your webcam, run the following command:

```bash
poetry run python src/testing/webcam_test.py
```

### 2. Test with the test dataset

To test the model using the test dataset, run the following command:

```bash
poetry run python src/testing/pictures_test.py
```
