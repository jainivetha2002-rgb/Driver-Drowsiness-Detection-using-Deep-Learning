Driver Drowsiness Detection using Deep Learning
Project Overview

Driver fatigue is a major cause of road accidents worldwide. Drowsiness reduces a driver’s alertness, reaction time, and decision-making ability. This project develops a computer vision–based deep learning system that detects driver fatigue by analyzing eye closure and yawning behavior from facial images.

The system classifies driver images into four categories and then maps them into three fatigue stages to provide an understandable fatigue status.

Problem Statement

Traditional driver monitoring systems often rely on wearable sensors or vehicle behavior, which can be intrusive or unreliable.

The goal of this project is to build a non-intrusive vision-based driver monitoring system that detects fatigue using physiological facial indicators such as:

Eye closure

Yawning behavior

Technologies Used

Python

TensorFlow / Keras

Convolutional Neural Networks (CNN)

Transfer Learning (MobileNetV2)

OpenCV / Image Processing

Streamlit (for web application)

Matplotlib & Seaborn (for visualization)

Project Workflow
1. Data Preparation

The dataset contains images of driver faces categorized into:

Eyes Open

Eyes Closed

Yawn

No Yawn

Images were preprocessed using:

Resizing to 224 × 224

Pixel normalization

Data augmentation:

Rotation

Zoom

Horizontal flipping

2. Model Development

Two models were implemented:

Custom CNN Model

A Convolutional Neural Network was built to extract features directly from facial images.

Transfer Learning Model

A MobileNetV2 pretrained model was used with custom classification layers to improve accuracy and generalization.

Transfer learning significantly improved performance compared to the custom CNN model.

3. Model Training

Key training settings:

Optimizer: Adam

Loss Function: Categorical Crossentropy

Input Size: 224 × 224

Training Strategy:

Freeze pretrained layers

Train classification head

4. Model Evaluation

The models were evaluated using:

Accuracy

Confusion Matrix

Training vs Validation Accuracy Graphs

MobileNetV2 achieved better performance compared to the custom CNN model.

Future Improvements

Possible enhancements include:

Real-time webcam-based fatigue detection

Integration with vehicle alert systems

Eye blink rate analysis

Deployment on embedded systems for real-time monitoring

Conclusion

This project demonstrates the effectiveness of deep learning and computer vision techniques in detecting driver fatigue. By combining CNN models with transfer learning and an interactive Streamlit interface, the system provides a practical and interpretable driver monitoring solution that can contribute to improving road safety.
