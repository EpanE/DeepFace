import os
import pickle
import numpy as np
import cv2
from deepface import DeepFace
from scipy.spatial.distance import cosine
import datetime
import time


def compile_face_embeddings(database_path='./employee_database',
                            output_file='face_embeddings.pkl'):
    """
    Compile face embeddings from a database directory into a single pickle file

    Args:
        database_path (str): Path to the database directory
        output_file (str): Path to save the compiled embeddings

    Returns:
        dict: Compiled face embeddings
    """
    known_embeddings = {}

    # Iterate through database directories
    for person_folder in os.listdir(database_path):
        folder_path = os.path.join(database_path, person_folder)

        if os.path.isdir(folder_path):
            try:
                # Find image files
                image_files = [f for f in os.listdir(folder_path)
                               if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

                if image_files:
                    # Use first image in the folder
                    image_path = os.path.join(folder_path, image_files[0])

                    # Extract embedding using Facenet512
                    embedding = DeepFace.represent(
                        img_path=image_path,
                        model_name='Facenet512',
                        enforce_detection=False,
                        detector_backend='retinaface'
                    )[0]['embedding']

                    # Store embedding if successful
                    known_embeddings[person_folder] = embedding
                    print(f"Loaded embedding for {person_folder}")

            except Exception as e:
                print(f"Error loading embedding for {person_folder}: {e}")

    # Save embeddings to file
    with open(output_file, 'wb') as f:
        pickle.dump(known_embeddings, f)

    print(f"Embeddings saved to {output_file}")
    return known_embeddings


class LiveFaceRecognitionSystem:
    def __init__(self,
                 embedding_file='face_embeddings.pkl',
                 recognition_threshold=0.3,
                 detection_interval=1,  # Interval in seconds between detection attempts
                 min_detection_cooldown=3  # Minimum time between consecutive detections
                 ):
        """
        Initialize the Live Face Recognition System

        Args:
            embedding_file (str): Path to the pickle file containing face embeddings
            recognition_threshold (float): Cosine distance threshold for face recognition
            detection_interval (int): Time interval between detection attempts
            min_detection_cooldown (int): Minimum cooldown time between detections
        """
        self.known_embeddings = {}
        self.recognition_threshold = recognition_threshold
        self._last_recognition_results = []
        self.detected_faces_dir = 'detected_faces'

        # New interval control parameters
        self.detection_interval = detection_interval
        self.min_detection_cooldown = min_detection_cooldown
        self.last_detection_time = 0
        self.next_detection_time = 0

        # Create directory for detected faces if it doesn't exist
        os.makedirs(self.detected_faces_dir, exist_ok=True)

        # Load pre-compiled embeddings
        try:
            with open(embedding_file, 'rb') as f:
                self.known_embeddings = pickle.load(f)
            print("Embeddings loaded successfully from compiled file")
        except FileNotFoundError:
            print("No compiled embeddings found. Create them first using compile_face_embeddings()")

    def is_detection_allowed(self):
        """
        Check if a new detection is allowed based on time intervals

        Returns:
            bool: Whether detection is allowed
        """
        current_time = time.time()

        # Check if enough time has passed since the last detection
        if current_time >= self.next_detection_time:
            # Update detection times
            self.last_detection_time = current_time
            self.next_detection_time = current_time + self.detection_interval
            return True

        return False

    def detect_faces(self, frame):
        """
        Detect faces in a frame

        Args:
            frame (numpy.ndarray): Input image frame

        Returns:
            bool: True if faces are detected, False otherwise
        """
        try:
            # Ensure image is in BGR color format for DeepFace
            if frame.shape[2] == 4:  # If RGBA, convert to BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

            # Use DeepFace representation to check for faces
            results = DeepFace.represent(
                img_path=frame,
                model_name='Facenet512',
                enforce_detection=False,
                detector_backend='retinaface'
            )

            return len(results) > 0
        except Exception as e:
            print(f"Error detecting faces: {e}")
            return False

    def recognize_faces_in_image(self, image):
        """
        Perform face recognition on a single image

        Args:
            image (numpy.ndarray): Image to analyze

        Returns:
            list: List of recognized faces with their names and confidence scores
        """
        try:
            # Reset last recognition results
            self._last_recognition_results = []

            # Ensure image is in BGR color format for DeepFace
            if image.shape[2] == 4:  # If RGBA, convert to BGR
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

            # Detect face embeddings
            recognition_results = DeepFace.represent(
                img_path=image,
                model_name='Facenet512',
                enforce_detection=False,
                detector_backend='retinaface'
            )

            # Compare each detected face with known embeddings
            for recognition_result in recognition_results:
                face_embedding = recognition_result['embedding']

                # Find the closest match
                best_match = None
                lowest_distance = float('inf')

                for name, known_embedding in self.known_embeddings.items():
                    # Calculate cosine distance
                    distance = cosine(face_embedding, known_embedding)

                    # Check if distance is below threshold
                    if distance < lowest_distance:
                        lowest_distance = distance
                        best_match = name

                # If a match is found below the threshold
                if best_match and lowest_distance <= self.recognition_threshold:
                    # Convert distance to a confidence score (0-1 range)
                    confidence = 1 - lowest_distance
                    result = {
                        'name': best_match,
                        'confidence': confidence,
                        'distance': lowest_distance
                    }
                else:
                    result = {
                        'name': 'Unknown',
                        'confidence': 0,
                        'distance': lowest_distance
                    }

                self._last_recognition_results.append(result)

            return self._last_recognition_results

        except Exception as e:
            print(f"Error in face recognition: {e}")
            return []

    def run_live_recognition(self, video_source=1):
        """
        Run live recognition system with automatic face detection

        Args:
            video_source (int): Camera index (default 1 for webcam)
        """
        # Open video capture
        cap = cv2.VideoCapture(video_source)

        # Window to display camera feed
        cv2.namedWindow('Live Camera', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Live Camera', 800, 600)

        # Flag to track face detection state
        face_detected = False
        frame_with_faces = None

        while True:
            # Read a frame from the video capture
            ret, frame = cap.read()

            if not ret:
                print("Failed to grab frame")
                break

            # Display the current frame
            cv2.imshow('Live Camera', frame)

            # Check for face detection with interval control
            if not face_detected and self.is_detection_allowed():
                face_detected = self.detect_faces(frame)

                if face_detected:
                    # Take a picture when face is first detected
                    frame_with_faces = frame.copy()
                    try:
                        # Perform face recognition on the captured frame
                        results = self.recognize_faces_in_image(frame_with_faces)

                        if results:
                            # Print detailed recognition information
                            print("\nRecognition Results:")
                            for result in results:
                                print(f"Name: {result['name']}")
                                print(f"Confidence: {result['confidence']:.2f}")
                                print(f"Distance: {result['distance']:.4f}")
                                print("---")

                        else:
                            print("No faces recognized.")

                    except Exception as e:
                        print(f"Error processing frame: {e}")

            # Wait for key press
            key = cv2.waitKey(1) & 0xFF

            # 'r' key pressed - reset face detection
            if key == ord('r'):
                face_detected = False
                frame_with_faces = None

            # 'q' key pressed - quit
            elif key == ord('q'):
                break

        # Release resources
        cap.release()
        cv2.destroyAllWindows()

    def get_last_recognition_results(self):
        """
        Retrieve the last recognition results

        Returns:
            list: Last recognition results
        """
        return self._last_recognition_results


# Usage example
if __name__ == '__main__':
    # First, compile embeddings (run this once to create the embedding file)
    compile_face_embeddings()

    # Initialize live recognition system with custom detection intervals
    recognition_system = LiveFaceRecognitionSystem(
        detection_interval=1,  # Seconds between detection attempts
        min_detection_cooldown=1  # Minimum time between detections
    )

    # Run live recognition
    recognition_system.run_live_recognition()