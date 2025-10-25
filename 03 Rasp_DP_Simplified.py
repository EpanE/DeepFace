import os
import pickle
import numpy as np
import cv2
from deepface import DeepFace
from scipy.spatial.distance import cosine
import datetime
import time
import csv
import tkinter as tk
from tkinter import ttk, simpledialog, messagebox
import threading
import platform
from PIL import Image, ImageTk  # Add PIL import for better image handling


# Previous functions remain the same
def compile_face_embeddings(database_path='./employee_database',
                            output_file='face_embeddings.pkl'):
    """
    Compile face embeddings from a database directory into a single pickle file
    """
    known_embeddings = {}
    os.makedirs(database_path, exist_ok=True)

    for person_folder in os.listdir(database_path):
        folder_path = os.path.join(database_path, person_folder)
        if os.path.isdir(folder_path):
            try:
                image_files = [f for f in os.listdir(folder_path)
                               if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if image_files:
                    image_path = os.path.join(folder_path, image_files[0])
                    embedding = DeepFace.represent(
                        img_path=image_path,
                        model_name='Facenet512',
                        enforce_detection=False,
                        detector_backend='retinaface'
                    )[0]['embedding']
                    known_embeddings[person_folder] = embedding
                    print(f"Loaded embedding for {person_folder}")
            except Exception as e:
                print(f"Error loading embedding for {person_folder}: {e}")

    with open(output_file, 'wb') as f:
        pickle.dump(known_embeddings, f)
    print(f"Embeddings saved to {output_file}")
    return known_embeddings


class LiveFaceRecognitionSystem:
    def __init__(self,
                 embedding_file='face_embeddings.pkl',
                 recognition_threshold=0.4,
                 attendance_log_path='attendance_log.csv'):
        """
        Initialize the Live Face Recognition System
        """
        self.known_embeddings = {}
        self.recognition_threshold = recognition_threshold
        self.attendance_log_path = attendance_log_path
        self.detected_faces_dir = 'detected_faces'

        os.makedirs(self.detected_faces_dir, exist_ok=True)
        os.makedirs('./employee_database', exist_ok=True)

        try:
            with open(embedding_file, 'rb') as f:
                self.known_embeddings = pickle.load(f)
            print("Embeddings loaded successfully")
        except FileNotFoundError:
            print("No compiled embeddings found")
            self.known_embeddings = compile_face_embeddings()

        if not os.path.exists(attendance_log_path):
            with open(attendance_log_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Name', 'Time'])

    def create_new_face_directory(self, name, num_photos=5):
        """
        Create a new directory for a person and capture their face photos
        """
        new_person_dir = os.path.join('./employee_database', name)
        os.makedirs(new_person_dir, exist_ok=True)

        cap = cv2.VideoCapture(0)
        photo_count = 0

        while photo_count < num_photos:
            ret, frame = cap.read()
            if ret:
                photo_path = os.path.join(new_person_dir, f'{name}_photo_{photo_count + 1}.jpg')
                cv2.imwrite(photo_path, frame)
                photo_count += 1

                cv2.putText(frame, f'Capturing photo {photo_count}/{num_photos}',
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Capturing Photos', frame)
                cv2.waitKey(1000)

        cap.release()
        cv2.destroyAllWindows()
        self.known_embeddings = compile_face_embeddings()

    def recognize_faces_in_image(self, image):
        """
        Perform face recognition on a single image
        """
        try:
            if image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

            results = DeepFace.represent(
                img_path=image,
                model_name='Facenet512',
                enforce_detection=False,
                detector_backend='retinaface'
            )

            if not results:
                return None

            face_embedding = results[0]['embedding']
            best_match = None
            lowest_distance = float('inf')

            for name, known_embedding in self.known_embeddings.items():
                distance = cosine(face_embedding, known_embedding)
                if distance < lowest_distance:
                    lowest_distance = distance
                    best_match = name

            if best_match and lowest_distance <= self.recognition_threshold:
                return best_match
            return "Unknown"

        except Exception as e:
            print(f"Error in face recognition: {e}")
            return None

    def run_live_recognition(self, video_source=0):
        root = tk.Tk()
        root.title("Attendance System")

        # Set window size to 480x320
        root.geometry("480x320")

        # Create main container
        main_frame = ttk.Frame(root)
        main_frame.pack(expand=True, fill='both')

        # Create left frame for info (1/3 of width)
        info_frame = ttk.Frame(main_frame, width=160)
        info_frame.pack(side=tk.LEFT, fill='y', padx=10, pady=10)
        info_frame.pack_propagate(False)

        # Name display
        name_label = ttk.Label(info_frame, text="Name:", font=("Helvetica", 12))
        name_label.pack(pady=(20, 0))
        name_display = ttk.Label(info_frame, text="---", font=("Helvetica", 14, "bold"))
        name_display.pack()

        # Time display
        time_label = ttk.Label(info_frame, text="Time:", font=("Helvetica", 12))
        time_label.pack(pady=(20, 0))
        time_display = ttk.Label(info_frame, text="--:--:--", font=("Helvetica", 14, "bold"))
        time_display.pack()

        # Buttons at bottom of info frame
        button_frame = ttk.Frame(info_frame)
        button_frame.pack(side=tk.BOTTOM, pady=20)

        register_button = ttk.Button(button_frame, text="Register Face")
        register_button.pack(pady=5)

        capture_button = ttk.Button(button_frame, text="Capture")
        capture_button.pack(pady=5)

        # Create right frame for video (2/3 of width)
        video_frame = ttk.Frame(main_frame)
        video_frame.pack(side=tk.LEFT, expand=True, fill='both', padx=10, pady=10)

        # Video display
        video_label = ttk.Label(video_frame)
        video_label.pack(expand=True)

        # Video capture
        cap = cv2.VideoCapture(video_source)

        def update_clock():
            """Update the time display"""
            current_time = datetime.datetime.now().strftime("%H:%M:%S")
            time_display.config(text=current_time)
            root.after(1000, update_clock)

        def update_frame():
            """Update the video frame"""
            ret, frame = cap.read()
            if ret:
                # Calculate aspect ratio to fit in available space
                video_width = 280  # Approximately 2/3 of 480 minus padding
                video_height = 210  # Maintains 4:3 aspect ratio

                # Resize frame
                frame = cv2.resize(frame, (video_width, video_height))

                # Convert BGR to RGB using PIL for correct color reproduction
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                photo = ImageTk.PhotoImage(image=image)

                # Update the label
                video_label.configure(image=photo)
                video_label.image = photo  # Keep a reference

            root.after(10, update_frame)

        def capture_and_recognize():
            """Capture frame and perform recognition"""
            ret, frame = cap.read()
            if ret:
                name = self.recognize_faces_in_image(frame)
                if name:
                    current_time = datetime.datetime.now()
                    name_display.config(text=name)

                    with open(self.attendance_log_path, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([name, current_time.strftime("%Y-%m-%d %H:%M:%S")])

        def register_new_face():
            """Register a new face"""
            name = simpledialog.askstring("Register New Face", "Enter name:")
            if name:
                threading.Thread(target=self.create_new_face_directory,
                                 args=(name,),
                                 daemon=True).start()

        # Bind buttons
        capture_button.config(command=capture_and_recognize)
        register_button.config(command=register_new_face)

        # Start updates
        update_clock()
        update_frame()

        root.mainloop()
        cap.release()


# Usage example
if __name__ == '__main__':
    recognition_system = LiveFaceRecognitionSystem()
    recognition_system.run_live_recognition()