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
from tkinter import ttk, simpledialog, messagebox
import threading
import platform

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
                 min_detection_cooldown=3,
                 attendance_log_path='attendance_log.csv'
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

        # Capture control parameters
        self.capture_countdown = 10
        self.is_capturing = False
        self.capture_timer = None

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
        self.attendance_log_path = attendance_log_path
        self.attendance_cooldown = timedelta(minutes=10)

        # Create CSV file with headers if it doesn't exist
        self._initialize_attendance_log()

        # Adjust for Raspberry Pi's lower computational power
        #if 'armv7l' in platform.machine():
        #     self.detection_interval = 3
        #     self.recognition_threshold = 0.4

        # Add a flag to track if recognition is currently allowed
        self.recognition_allowed = True

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

    def create_new_face_directory(self, name, num_photos=50):
        """
        Create a new directory for a person and capture their face photos

        Args:
            name (str): Name of the person
            num_photos (int): Number of photos to capture
        """
        # Create directory for the new person
        new_person_dir = os.path.join('./employee_database', name)
        os.makedirs(new_person_dir, exist_ok=False)

        def capture_photos(cap, directory, name, total_photos):
            photo_count = 0
            while photo_count < total_photos:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break

                # Save the frame
                photo_path = os.path.join(directory, f'{name}_photo_{photo_count + 1}.jpg')
                cv2.imwrite(photo_path, frame)
                photo_count += 1

                # Show countdown
                cv2.putText(frame, f'Capturing photo {photo_count}/{total_photos}',
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Capturing Photos', frame)

                # Wait between photos
                cv2.waitKey(1000)  # 1 second between photos

            cap.release()
            cv2.destroyAllWindows()

            # Compile embeddings after capturing
            compile_face_embeddings()

        # Open video capture
        cap = cv2.VideoCapture(0)

        # Start photo capturing in a separate thread
        threading.Thread(target=capture_photos,
                         args=(cap, new_person_dir, name, num_photos),
                         daemon=True).start()

    def run_live_recognition(self, video_source=1):
        root = tk.Tk()
        root.title("Live Face Recognition")

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

        # Countdown Label
        countdown_label = ttk.Label(main_frame, text="", font=("Helvetica", 12))
        countdown_label.grid(row=2, column=0, columnspan=2, pady=10)

        # Attendance Logging
        attendance_logging_frame = ttk.LabelFrame(main_frame, text="Attendance Logging")
        attendance_logging_frame.grid(row=3, column=0, columnspan=2, padx=10, pady=10)

        attendance_logging_text = tk.Text(attendance_logging_frame, width=80, height=5)
        attendance_logging_text.pack()

        cap = cv2.VideoCapture(video_source)

        # Store the current frame for later use in recognition
        current_frame = None

        def start_capture_countdown(self):
            """Start 10-second countdown before next capture"""
            self.is_capturing = False
            self.recognition_allowed = False  # Disable recognition during countdown

            if hasattr(self, 'capture_timer'):
                root.after_cancel(self.capture_timer)

            def countdown():
                if self.capture_countdown > 0:
                    # Update countdown on UI instead of printing
                    countdown_label.config(text=f"Next capture in {self.capture_countdown} seconds")
                    self.capture_countdown -= 1
                    self.capture_timer = root.after(1000, countdown)
                else:
                    # Reset state more explicitly
                    self.capture_countdown = 10
                    self.is_capturing = False
                    self.recognition_allowed = True  # Re-enable recognition
                    countdown_label.config(text="")  # Clear countdown label

            countdown()

        def on_key_press(event):
            """Handle key press events for face recognition"""
            if event.char == 'q' and not self.is_capturing and current_frame is not None:
                # More robust recognition allowance check
                if not self.recognition_allowed:
                    messagebox.showinfo("Wait", "Please wait for the next capture window")
                    return

                try:
                    # Perform face recognition on the current frame
                    results = self.recognize_faces_in_image(current_frame)

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

                    # Start capture countdown
                    start_capture_countdown()

                except Exception as e:
                    print(f"Error processing frame: {e}")
                    # Reset recognition state in case of error
                    self.recognition_allowed = True
                    self.is_capturing = False


        root.bind('q', on_key_press)

        def update_frame():
            """Update the video frame and display"""
            nonlocal current_frame

            # Read a frame from the video capture
            ret, frame = cap.read()

            if not ret:
                print("Failed to grab frame")
                return

            # Store the current frame for recognition
            current_frame = frame.copy()

            # Resize the image to fit the label
            img = cv2.resize(frame, (320, 240))

            # Convert the image to PhotoImage
            photo = tk.PhotoImage(data=cv2.imencode('.png', img)[1].tobytes())
            live_video_label.config(image=photo)
            live_video_label.image = photo

            # Schedule the next frame update
            root.after(10, update_frame)


        # Add a new face registration button
        def new_face_registration():
            name = simpledialog.askstring("New Face", "Enter name for new person:")
            if name:
                try:
                    self.create_new_face_directory(name)
                    messagebox.showinfo("Success", f"Created directory and started capturing photos for {name}")
                except Exception as e:
                    messagebox.showerror("Error", str(e))

        new_face_button = ttk.Button(main_frame, text="Register New Face", command=new_face_registration)
        new_face_button.grid(row=4, column=0, pady=10)

        # Add a quit button
        quit_button = ttk.Button(main_frame, text="Quit", command=root.destroy)
        quit_button.grid(row=4, column=1, pady=10)

        # Start the frame update loop
        update_frame()

        root.mainloop()

        # Release resources after the window is closed
        cap.release()
        cv2.destroyAllWindows()

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

# Usage example
if __name__ == '__main__':
    # First, compile embeddings (run this once to create the embedding file)
    compile_face_embeddings()

    # Initialize live recognition system
    recognition_system = LiveFaceRecognitionSystem(
        detection_interval=2,
        min_detection_cooldown=2,
        attendance_log_path='attendance_log.csv'
    )

    # Run live recognition
    recognition_system.run_live_recognition()