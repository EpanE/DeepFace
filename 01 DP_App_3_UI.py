import os
import pickle
import numpy as np
import cv2
from deepface import DeepFace
from scipy.spatial.distance import cosine
import datetime
import time
import csv
from datetime import timedelta
import tkinter as tk
from tkinter import ttk


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
                 recognition_threshold=0.4,
                 detection_interval=3,  # Interval in seconds between detection attempts
                 min_detection_cooldown=3,  # Minimum time between consecutive detections
                 attendance_log_path='attendance_log.csv'  # Add this line
                 ):
        """
        Initialize the Live Face Recognition System

        Args:
            embedding_file (str): Path to the pickle file containing face embeddings
            recognition_threshold (float): Cosine distance threshold for face recognition
            detection_interval (int): Time interval between detection attempts
            min_detection_cooldown (int): Minimum cooldown time between detections
            attendance_log_path (str): Path to the attendance log file
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

        # Attendance logging setup
        self.attendance_log_path = attendance_log_path  # Update this line
        self.attendance_cooldown = timedelta(minutes=10)

        # Create CSV file with headers if it doesn't exist
        self._initialize_attendance_log()

    def _initialize_attendance_log(self):
        """
        Initialize the attendance log CSV file with headers if it doesn't exist
        """
        if not os.path.exists(self.attendance_log_path):
            with open(self.attendance_log_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Name', 'Date', 'Time'])

    def log_attendance(self, name):
        """
        Log attendance for a recognized face with 10-minute cooldown

        Args:
            name (str): Name of the recognized person
        """
        if name == 'Unknown':
            return

        current_time = datetime.datetime.now()

        # Check if the person has been logged recently
        try:
            with open(self.attendance_log_path, 'r') as csvfile:
                reader = csv.reader(csvfile)
                next(reader)  # Skip header

                # Reverse the log to check recent entries first
                log_entries = list(reader)
                for entry in reversed(log_entries):
                    if entry[0] == name:
                        # Parse the previous log entry time
                        prev_log_time = datetime.datetime.strptime(
                            f"{entry[1]} {entry[2]}", "%Y-%m-%d %H:%M:%S"
                        )

                        # Check if less than 10 minutes have passed
                        if current_time - prev_log_time < self.attendance_cooldown:
                            return  # Skip logging if within cooldown period
                        break

        except (FileNotFoundError, IndexError):
            pass  # First-time logging or file issues

        # Log the attendance
        try:
            with open(self.attendance_log_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    name,
                    current_time.strftime("%Y-%m-%d"),
                    current_time.strftime("%H:%M:%S")
                ])
            print(f"Attendance logged for {name}")
        except Exception as e:
            print(f"Error logging attendance: {e}")

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

    def run_live_recognition(self, video_source=0):
        root = tk.Tk()
        root.title("Live Face Recognition")

        import platform

        # Adjust for Raspberry Pi's lower computational power
        if 'armv7l' in platform.machine():
            self.detection_interval = 3
            self.recognition_threshold = 0.4

        # Create the UI layout
        main_frame = ttk.Frame(root)
        main_frame.pack(padx=20, pady=20)

        # Live Video Feed
        live_video_frame = ttk.LabelFrame(main_frame, text="Live Video Feed")
        live_video_frame.grid(row=0, column=0, padx=10, pady=10)

        live_video_label = ttk.Label(live_video_frame)
        live_video_label.pack()

        # Face Detection History
        detection_history_frame = ttk.LabelFrame(main_frame, text="History of Detection")
        detection_history_frame.grid(row=0, column=1, padx=10, pady=10)

        detection_history_text = tk.Text(detection_history_frame, width=40, height=10)
        detection_history_text.pack()

        # Attendance Logging
        attendance_logging_frame = ttk.LabelFrame(main_frame, text="Attendance Logging")
        attendance_logging_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=10)

        attendance_logging_text = tk.Text(attendance_logging_frame, width=80, height=5)
        attendance_logging_text.pack()

        cap = cv2.VideoCapture(video_source)

        def update_frame():
            # Read a frame from the video capture
            ret, frame = cap.read()

            if not ret:
                print("Failed to grab frame")
                return

            # Option 1: Keep original BGR format
            img = frame

            # Or Option 2: Explicit RGB conversion if needed
            # img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Resize the image to fit the label
            img = cv2.resize(img, (320, 240))

            # Convert the image to PhotoImage
            photo = tk.PhotoImage(data=cv2.imencode('.png', img)[1].tobytes())
            live_video_label.config(image=photo)
            live_video_label.image = photo

            # Check for face detection with interval control
            if self.is_detection_allowed():
                try:
                    # Detect faces
                    face_detected = self.detect_faces(frame)

                    if face_detected:
                        # Perform face recognition on the captured frame
                        results = self.recognize_faces_in_image(frame)

                        if results:
                            # Update the detection history text box
                            detection_history_text.delete("1.0", tk.END)
                            for result in results:
                                # Log attendance for recognized faces
                                self.log_attendance(result['name'])
                                detection_history_text.insert(tk.END,
                                                              f"Name: {result['name']}, Confidence: {result['confidence']:.2f}, Distance: {result['distance']:.4f}\n")

                            # Update the attendance logging text box
                            with open(self.attendance_log_path, 'r') as csvfile:
                                reader = csv.reader(csvfile)
                                next(reader)  # Skip header
                                attendance_logging_text.delete("1.0", tk.END)
                                for row in reader:
                                    attendance_logging_text.insert(tk.END, f"{row[0]}, {row[1]}, {row[2]}\n")
                        else:
                            detection_history_text.delete("1.0", tk.END)
                            detection_history_text.insert(tk.END, "No faces recognized.")

                except Exception as e:
                    print(f"Error processing frame: {e}")

            # Schedule the next frame update
            root.after(10, update_frame)

        # Start the frame update loop
        update_frame()

        # Add a quit button
        quit_button = ttk.Button(main_frame, text="Quit", command=root.destroy)
        quit_button.grid(row=2, column=0, columnspan=2, pady=10)

        root.mainloop()

        # Release resources after the window is closed
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
        detection_interval=2,
        min_detection_cooldown=2,
        attendance_log_path='attendance_log.csv'
    )

    # Run live recognition
    recognition_system.run_live_recognition()