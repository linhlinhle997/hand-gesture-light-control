import os
import cv2
import torch
import numpy as np
from generate_landmark_data import HandLandmarksDetector
from custom_nn_utils import NeuralNetwork
from generate_landmark_data import label_dict_from_config_file

MODEL_PATH = "models"


class LightGesture:
    def __init__(self, model_path, device=False):
        self.device = device
        self.height, self.width = 720, 1280 # Camera frame dimensions

        # Create hand gesture detector object to process images and detect hands
        self.detector = HandLandmarksDetector()
        self.status_text = None # Text that shows the current action or status
        self.signs = label_dict_from_config_file("hand_gesture.yaml") # Load gesture labels from config file
        self.classifier = self.load_model(model_path) # Load the pre-trained neural network model

        # Initial state for lights (off by default)
        self.light1, self.light2, self.light3 = False, False, False

    def load_model(self, model_path):
        """Load the neural network model."""
        classifier = NeuralNetwork()
        state_dict = torch.load(model_path, weights_only=True) # Load pre-trained model weights
        classifier.load_state_dict(state_dict) # Apply the weights to the model
        classifier.eval() # Set the model to evaluation mode (disable dropout, etc.)
        return classifier

    def light_device(self, img, lights):
        """Draw the light status panel on the image."""
        height, width, _ = img.shape # Get the dimensions of the input image
        rect_height = int(0.15 * height) # Set height of the light status panel
        rect_width = width # Set width to match image width
        white_rect = np.ones((rect_height, rect_width, 3), dtype=np.uint8) * 255 # Create a white rectangle for the panel

        # Draw circles representing light states (on or off)
        circle_radius = int(0.45 * rect_height)
        positions = [
            (int(width * 0.25), int(rect_height / 2)),  # Light 1 position
            (int(width * 0.5), int(rect_height / 2)), # Light 2 position
            (int(width * 0.75), int(rect_height / 2)), # Light 3 position
        ]
        # Colors depend on whether the light is on (yellow) or off (black)
        colors = [(0, 255, 255) if light else (0, 0, 0) for light in lights]

        # Draws circles at specified positions with corresponding colors
        for position, color in zip(positions, colors):
            cv2.circle(white_rect, position, circle_radius, color, -1)
        
        # Stack the light status panel below the image
        img = np.vstack((img, white_rect))
        return img

    def run(self):
        """Main loop for gesture detection and light control."""
        cam = cv2.VideoCapture(0)
        cam.set(3, 1280)
        cam.set(4, 720)

        while cam.isOpened():
            _, frame = cam.read()
            hand, img = self.detector.detect_hand(frame) # Detects hands in the frame
            
            # If hands are detected
            if len(hand) != 0:
                with torch.no_grad():
                    # Flatten the detected hand landmarks and convert them to a tensor
                    hand_landmark = torch.from_numpy(np.array(hand[0], dtype=np.float32).flatten()).unsqueeze(0)
                    # Predict the gesture class from the neural network model
                    class_number = self.classifier.predict(hand_landmark).item()
                    
                    # If a valid gesture is recognized
                    if class_number != -1:
                        # Get the gesture name from the label dictionary
                        gesture = self.signs[class_number]

                        # Handle gesture actions to control lights
                        if gesture  == "light1" and not self.light1:
                            self.light1 = True
                            self.status_text = "Turning on light 1"
                        elif gesture  == "light2" and not self.light2:
                            self.light2 = True
                            self.status_text = "Turning on light 2"
                        elif gesture  == "light3" and not self.light3:
                            self.light3 = True
                            self.status_text = "Turning on light 3"
                        elif gesture  == "turn_on":
                            self.light1 = self.light2 = self.light3 = True
                            self.status_text = "Turning on all lights"
                        elif gesture  == "turn_off":
                            self.light1 = self.light2 = self.light3 = False
                            self.status_text = "Turning off all lights"
                        else:
                            # If an unrecognized gesture is detected
                            self.status_text = "undefined command"
                    else:
                        # If no valid gesture is recognized
                        self.status_text = "undefined command"
            else:
                # Resets the status text if no hand is detected
                self.status_text = None

            # Update the image with the current light status
            img = self.light_device(img, [self.light1, self.light2, self.light3])

            # Displays the status text on the image
            if self.status_text:
                cv2.putText(img, self.status_text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Create a window to display the video feed
            cv2.namedWindow("window", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("window", 1024, 768) # Resize window for display
            cv2.imshow("window", img) # Show the updated image in the window

            if cv2.waitKey(1) == ord("q"):
                break
            
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    model_path = os.path.join(MODEL_PATH, "best_model.pth")
    light = LightGesture(model_path, device=True)
    light.run()
