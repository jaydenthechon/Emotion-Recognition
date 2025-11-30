# Emotion Recognition using Webcam

A real-time emotion detection system using OpenCV and deep learning that recognizes 7 emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.

## Features

- Real-time face detection using webcam
- Emotion classification with confidence scores
- CNN-based deep learning model
- Support for 7 different emotions

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Make sure you have a webcam connected to your computer.

## Usage

### Quick Start (without pre-trained model)

Run the emotion detector (will only show face detection without a model):
```bash
python emotion_detection.py
```

Press 'q' to quit the application.

### Using a Pre-trained Model

To get full emotion recognition, you need a trained model:

#### Option 1: Download a pre-trained model
1. Search for "FER2013 emotion detection model" on GitHub or Kaggle
2. Download the `.h5` model file
3. Place it in the project directory as `emotion_model.h5`
4. Run `python emotion_detection.py`

#### Option 2: Train your own model
1. Download the FER2013 dataset from Kaggle
2. Organize the data into this structure:
   ```
   data/
   ├── train/
   │   ├── angry/
   │   ├── disgust/
   │   ├── fear/
   │   ├── happy/
   │   ├── sad/
   │   ├── surprise/
   │   └── neutral/
   └── validation/
       ├── angry/
       ├── disgust/
       ├── fear/
       ├── happy/
       ├── sad/
       ├── surprise/
       └── neutral/
   ```
3. Run the training script:
   ```bash
   python train_model.py
   ```
4. After training completes, run the detector:
   ```bash
   python emotion_detection.py
   ```

## How It Works

1. **Face Detection**: Uses Haar Cascade classifier to detect faces in video frames
2. **Preprocessing**: Converts detected faces to 48x48 grayscale images
3. **Emotion Classification**: CNN model predicts emotion from facial features
4. **Display**: Shows emotion label with confidence percentage

## Model Architecture

The CNN model includes:
- 3 convolutional blocks with batch normalization
- MaxPooling and Dropout layers for regularization
- Dense layers for classification
- Softmax output for 7 emotion classes

## Troubleshooting

**Webcam not working:**
- Check if your webcam is properly connected
- Try changing the camera index in `cv2.VideoCapture(0)` to `1` or `2`

**Model not loading:**
- Ensure `emotion_model.h5` is in the project directory
- The application will run with face detection only (no emotion recognition)

**Low accuracy:**
- Train with more epochs
- Use data augmentation (already included in training script)
- Try different lighting conditions for better face detection

## Dataset

The model is designed to work with the FER2013 dataset:
- 35,887 grayscale images (48x48 pixels)
- 7 emotion categories
- Available on Kaggle: https://www.kaggle.com/datasets/msambare/fer2013

## Requirements

- Python 3.8+
- OpenCV
- TensorFlow/Keras
- NumPy
- Webcam

## License

This project is for educational purposes.
