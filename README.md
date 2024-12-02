# Hand Gesture Controlled Lights (Virtual Mode)

This project provides a virtual system for controlling lights using hand gestures. By leveraging Google's MediaPipe Gesture Recognizer and a Multi-Layer Perceptron (MLP) model, the system recognizes predefined hand gestures and executes corresponding light control commands—all in a simulated environment without requiring hardware.

## Project Overview

The system simulates three lights, toggling them based on recognized gestures. This project is ideal for exploring gesture recognition and virtual device control, requiring only a webcam for real-time gesture capture and simulation.

## Project Workflow

### 1. Data Preparation
- Data Collection:
    - Run the `generate_landmark_data.py` script to record hand gestures.
    - Define gesture classes in the `hand_gesture.yaml` configuration file.
    - Use specific keyboard keys to label and record gestures, saving data as CSV files.
- Feature Extraction:
    - MediaPipe extracts hand landmarks from the recorded gestures.
    - Data is split into training, validation, and test sets for model development.

### 2. Model Training
- Model Definition:
    - Use `train_model.ipynb` to define and train an MLP model to classify hand gestures.
- Training and Evaluation:
    - Train the model using the training and validation datasets.
    - Evaluate its accuracy and performance on the test dataset.

### 3. Real-Time System Deployment
- Real-Time Recognition:
    - Use `detect_simulation.py` to process real-time webcam input, recognize gestures, and perform virtual light control actions.
- Simulated Light Control:
    - Lights toggle visually on the webcam feed, simulating real-world light control.


## How to Run
Follow these steps to set up and execute the project:
### 1. Clone the repository:
Get a local copy of the project:
``` sh
git clone https://github.com/linhlinhle997/hand-gesture-light-control.git
cd hand-gesture-light-control
```
### 2. Install dependencies:
Ensure all required libraries are installed:
``` sh
pip install -r requirements.txt
```
### 3. Prepare data:
Use the provided script to collect and save hand gesture data:
1. Run the script:
    ``` sh
    python generate_landmark_data.py
    ```
2. Follow the on-screen instructions to record and label gesture data.
### 4. Train the model:
Train the MLP model to recognize hand gestures:
1. Open the `train_model.ipynb` notebook in your preferred Python environment.
2. Execute the notebook cells to define, train, and evaluate the model.
### 5. Simulate light control:
Activate the virtual light control system:
```sh
python detect_simulation.py
```
The system will recognize hand gestures in real time and simulate toggling lights on the webcam feed.

## Key Files
- `generate_landmark_data.py`: Captures hand gesture data and stores it in CSV format.
- `train_model.ipynb`: Defines and trains the MLP model.
- `detect_simulation.py`: Recognizes gestures and simulates light control actions.
- `hand_gesture.yaml`: Configuration file for defining gesture classes.

## Technologies Used
- `MediaPipe`: For real-time hand gesture recognition.
- `Python`: Programming language for all scripts.
- `Deep Learning (MLP)`: For gesture classification
