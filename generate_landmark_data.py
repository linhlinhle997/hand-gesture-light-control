import os
import csv
import cv2
import yaml
import numpy as np
import mediapipe as mp
from typing import Dict, Any


def is_handsign_character(char: str) -> bool:
    """Check if a character is a valid hand sign character."""
    return ord("a") <= ord(char) < ord("q") or char == " "


def label_dict_from_config_file(relative_path: str) -> Dict[int, str]:
    """Load the label dictionary from a YAML configuration file."""
    with open(relative_path, "r") as f:
        label_tag = yaml.full_load(f)["gestures"]
    return label_tag


class HandDatasetWriter():
    """Write hand gesture data (hand and label) to a CSV file."""
    def __init__(self, filepath: str) -> None:
        self.csv_file = open(filepath, "a")
        self.file_writer = csv.writer(
            self.csv_file, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )

    def add(self, hand: Any, label: int) -> None:
        """Adds a new row to the CSV file with hand data and its associated label."""
        self.file_writer.writerow([label, *np.array(hand).flatten().tolist()])

    def close(self) -> None:
        """Closes the opened CSV file."""
        self.csv_file.close()


class HandLandmarksDetector():
    """Detect hand landmarks in a given frame using MediaPipe."""
    def __init__(self) -> None:
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands
        self.detector = self.mp_hands.Hands(
            False, max_num_hands=1, min_detection_confidence=0.5
        )

    def detect_hand(self, frame):
        """Detects hands and their landmarks in a given frame."""
        hands = []

        # Flip the frame horizontally for mirror-like effect.
        frame = cv2.flip(frame, 1)

        annotated_image = frame.copy()
        results = self.detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # If hand landmarks are found, process each hand's landmarks.
        if results.multi_hand_landmarks is not None:
            for hand_landmarks in results.multi_hand_landmarks:
                hand = []

                # Draw landmarks and connections on the annotated image.
                self.mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style(),
                )

                # Extract the 3D coordinates (x, y, z) of each landmark.
                for landmark in hand_landmarks.landmark:
                    x, y, z = landmark.x, landmark.y, landmark.z
                    hand.extend([x, y, z])
            hands.append(hand)
        return hands, annotated_image


def run(data_path, sign_img_path, split="val", resolution=(1280, 720)):
    hand_detector = HandLandmarksDetector()  # Initialize hand landmark detector
    cam = cv2.VideoCapture(0)  # Open default camera
    cam.set(3, resolution[0])  # Set camera width
    cam.set(4, resolution[1])  # Set camera height
    
    os.makedirs(data_path, exist_ok=True)  # Create data directory if not exists
    os.makedirs(sign_img_path, exist_ok=True)  # Create image directory if not exists
    print(sign_img_path)
    
    dataset_path = f"./{data_path}/landmark_{split}.csv"  # Generate dataset CSV path
    hand_dataset = HandDatasetWriter(dataset_path)  # Initialize dataset writer

    current_letter = None  # Track currently selected letter
    status_text = None  # Display status message
    cannot_switch_char = False  # Flag to prevent character switching
    saved_frame = None  # Store frame when hand sign is captured

    while cam.isOpened():
        _, frame = cam.read()  # Read camera frame
        hands, annotated_image = hand_detector.detect_hand(frame)  # Detect hands in frame

        # Set status text based on recording state
        if current_letter is None:
            status_text = "Press a character to record"
        else:
            # Convert letter to numeric label
            label = ord(current_letter) - ord("a")  
            # Handle special case for space key
            if label == -65:
                status_text = "Recording unknown, press spacebar again to stop"
                label = -1
            else:
                status_text = f"Recording {LABEL_TAG[label]}, press {current_letter} again to stop"

        key = cv2.waitKey(1)  # Wait for key press
        if key == -1:
            if current_letter is None:
                pass
            else:
                if len(hands) != 0:
                    hand = hands[0]
                    hand_dataset.add(hand=hand, label=label)  # Add hand landmarks to dataset
                    saved_frame = frame  # Save current frame
        else:
            key = chr(key)
            if key == "q":
                break  # Exit program

            # Handle character selection and recording
            if is_handsign_character(key):
                if current_letter is None:
                    current_letter = key
                elif current_letter == key:
                    if saved_frame is not None:
                        if label >= 0:
                            # Save hand sign image
                            cv2.imwrite(f"./{sign_img_path}/{LABEL_TAG[label]}.jpg", saved_frame)

                        # Reset recording state
                        cannot_switch_char = False
                        current_letter = None
                        saved_frame = None
                    else:
                        cannot_switch_char = True

        # Display message if cannot switch characters
        if cannot_switch_char:
            cv2.putText(annotated_image, f"please press {current_letter} again to unbind", (0, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        cv2.putText(annotated_image, status_text, (5,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow(f"{split}", annotated_image)

    cv2.destroyAllWindows()  # Close all OpenCV windows


if __name__ == "__main__":
    LABEL_TAG = label_dict_from_config_file("hand_gesture.yaml")
    data_path = "./data2"
    sign_img_path = "./sign_imgs2"
    run(data_path, sign_img_path, "train", (1280, 720))
    run(data_path, sign_img_path, "val", (1280, 720))
    run(data_path, sign_img_path, "test", (1280, 720))
