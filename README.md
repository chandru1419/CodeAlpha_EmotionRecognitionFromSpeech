# Emotion Recognition from Speech (Task 2 – CodeAlpha)

## Project Description
This project focuses on recognizing human emotions from speech audio using speech signal processing and deep learning techniques. The system classifies emotions such as happy, sad, angry, and neutral by analyzing speech signals. This project was completed as part of the Machine Learning Internship at CodeAlpha.

## Objective
The objective of this project is to build a deep learning–based emotion recognition model that can accurately predict human emotions from speech audio.

## Approach
Speech signals are processed using Mel-Frequency Cepstral Coefficients (MFCCs), which capture essential audio features related to human emotions. These features are then used to train deep learning models such as Convolutional Neural Networks (CNN) or Long Short-Term Memory (LSTM) networks for emotion classification.

## Tools and Technologies
This project is implemented using Python. Librosa is used for audio feature extraction, NumPy for numerical computations, TensorFlow and Keras for building deep learning models, and Scikit-learn for preprocessing and evaluation.

## Dataset
The project uses the RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset, which is a widely used dataset for speech emotion recognition. Due to GitHub file size limitations, the dataset is not included in this repository. It can be downloaded from the following link:  
https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio  

After downloading, the dataset should be extracted and placed in the project directory using the following structure:  
ravdess-emotional-speech-audio/Audio_Speech_Actors_01-24/

## Methodology
The workflow of the project includes loading speech audio files, extracting MFCC features, encoding emotion labels, splitting the dataset into training and testing sets, training a deep learning model, and evaluating the model using standard performance metrics.

## Model Evaluation
The model performance is evaluated using accuracy, precision, recall, F1-score, and a classification report to measure how effectively the model recognizes different emotions.

## Results
The trained deep learning model successfully recognizes emotions from speech audio and achieves reliable accuracy on the RAVDESS dataset, demonstrating the effectiveness of MFCC-based feature extraction combined with deep learning techniques.

## How to Run
Install the required dependencies using the following command:  
pip install librosa numpy scikit-learn tensorflow  

Download and extract the RAVDESS dataset as instructed above, then run the Python script or Jupyter Notebook included in this repository.

## Summary
An emotion recognition system was developed using MFCC-based speech feature extraction and deep learning models to classify human emotions from speech audio. The system was trained and evaluated on the RAVDESS dataset and demonstrated effective emotion prediction performance.

## Author
Chandru  
Machine Learning Intern – CodeAlpha

