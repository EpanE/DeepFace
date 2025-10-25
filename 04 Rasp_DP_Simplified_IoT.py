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
from PIL import Image, ImageTk
import requests
from urllib.parse import quote

model = 'Facenet'  # Can be Facenet / Facenet512 / OpenFace / etc
backend = 'retinaface'  # Don't change this


class LiveFaceRecognitionSystem:
    def __init__(self,
                 embedding_file='face_embeddings.pkl',
                 recognition_threshold=0.4,
                 attendance_log_path='attendance_log.csv',
                 google_script_url='https://script.google.com/macros/s/AKfycbx3Hu6trKN8hWiKKctnX5ei9-9ME1USuEV5mExr1L7xZiL_zqMAFbgjX7kMqxTIlL7S/exec'):
        """
        Initialize the Live Face Recognition System
        """
        self.known_embeddings = {}
        self.recognition_threshold = recognition_threshold
        self.attendance_log_path = attendance_log_path
        self.detected_faces_dir = 'detected_faces'
        self.google_script_url = google_script_url
        self.process_time = 0.0
        self.screen_width = 480
        self.screen_height = 320

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
                writer.writerow(['Name', 'Time', 'Process Time (ms)'])

    def log_to_google_sheets(self, name, timestamp):
        """
        Log attendance to Google Sheets via Google Apps Script
        """
        try:
            # Construct the URL with parameters
            encoded_name = quote(name)
            timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
            encoded_timestamp = quote(timestamp_str)

            url = f"{self.google_script_url}?name={encoded_name}&timestamp={encoded_timestamp}&processTime={self.process_time}&model={model}&backend={backend}"

            # Send the GET request
            response = requests.get(url)

            if response.status_code == 200:
                print("Successfully logged to Google Sheets")
            else:
                print(f"Failed to log to Google Sheets: {response.status_code}")

        except Exception as e:
            print(f"Error logging to Google Sheets: {e}")

    def recognize_faces_in_image(self, image):
        """
        Perform face recognition on a single image
        """
        start_time = time.time()
        try:
            if image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

            results = DeepFace.represent(
                img_path=image,
                model_name=model,
                enforce_detection=False,
                detector_backend=backend
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

            self.process_time = (time.time() - start_time) * 1000  # Convert to milliseconds

            if best_match and lowest_distance <= self.recognition_threshold:
                return best_match
            return "Unknown"

        except Exception as e:
            print(f"Error in face recognition: {e}")
            self.process_time = (time.time() - start_time) * 1000
            return None

    def create_new_face_directory(self, name):
        """
        Create a new directory for a person and capture their face
        """
        dir_path = os.path.join('./employee_database', name)
        os.makedirs(dir_path, exist_ok=True)

        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()

        if ret:
            # Save the captured image
            img_path = os.path.join(dir_path, f'{name}.jpg')
            cv2.imwrite(img_path, frame)
            messagebox.showinfo("Success", f"Face registered for {name}")

            # Recompile embeddings
            self.known_embeddings = compile_face_embeddings()
        else:
            messagebox.showerror("Error", "Failed to capture image")

    def run_live_recognition(self, video_source=0):
        root = tk.Tk()
        root.title("Attendance System")

        # Force the window size and make it non-resizable
        root.geometry(f"{self.screen_width}x{self.screen_height}")
        root.resizable(False, False)

        # Configure the grid
        root.grid_columnconfigure(0, weight=1)
        root.grid_columnconfigure(1, weight=2)

        # Create styles for smaller fonts
        style = ttk.Style()
        style.configure('Small.TLabel', font=('Helvetica', 8))
        style.configure('SmallBold.TLabel', font=('Helvetica', 9, 'bold'))
        style.configure('Small.TButton', font=('Helvetica', 8))

        # Left frame for info (160px width)
        info_frame = ttk.Frame(root, width=160)
        info_frame.grid(row=0, column=0, sticky='nsew', padx=2, pady=2)
        info_frame.grid_propagate(False)

        # Configure info frame grid
        info_frame.grid_columnconfigure(0, weight=1)
        for i in range(8):
            info_frame.grid_rowconfigure(i, weight=1)

        # Name display
        ttk.Label(info_frame, text="Name:", style='Small.TLabel').grid(row=0, pady=2)
        name_display = ttk.Label(info_frame, text="---", style='SmallBold.TLabel')
        name_display.grid(row=1, pady=0)

        # Time display
        ttk.Label(info_frame, text="Time:", style='Small.TLabel').grid(row=2, pady=2)
        time_display = ttk.Label(info_frame, text="--:--:--", style='SmallBold.TLabel')
        time_display.grid(row=3, pady=0)

        # Process Time display
        ttk.Label(info_frame, text="Process Time:", style='Small.TLabel').grid(row=4, pady=2)
        process_time_display = ttk.Label(info_frame, text="--- ms", style='SmallBold.TLabel')
        process_time_display.grid(row=5, pady=0)

        # Buttons
        button_frame = ttk.Frame(info_frame)
        button_frame.grid(row=7, pady=5)

        register_button = ttk.Button(button_frame, text="Register Face", style='Small.TButton', width=15)
        register_button.pack(pady=2)

        capture_button = ttk.Button(button_frame, text="Capture", style='Small.TButton', width=15)
        capture_button.pack(pady=2)

        # Right frame for video (320px width)
        video_frame = ttk.Frame(root)
        video_frame.grid(row=0, column=1, sticky='nsew', padx=2, pady=2)

        video_label = ttk.Label(video_frame)
        video_label.pack(expand=True, fill='both')

        # Initialize video capture
        cap = cv2.VideoCapture(video_source)

        def update_clock():
            current_time = datetime.datetime.now().strftime("%H:%M:%S")
            time_display.config(text=current_time)
            root.after(1000, update_clock)

        def update_frame():
            ret, frame = cap.read()
            if ret:
                # Calculate the video display size to maintain aspect ratio
                frame_height, frame_width = frame.shape[:2]
                aspect_ratio = frame_width / frame_height

                # Target width is 320px (screen width - info panel width)
                target_width = 310
                target_height = int(target_width / aspect_ratio)

                # Ensure the height doesn't exceed the screen height
                if target_height > self.screen_height:
                    target_height = self.screen_height - 10
                    target_width = int(target_height * aspect_ratio)

                # Resize the frame
                frame = cv2.resize(frame, (target_width, target_height))
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                photo = ImageTk.PhotoImage(image=image)

                video_label.configure(image=photo)
                video_label.image = photo

            root.after(10, update_frame)

        def capture_and_recognize():
            ret, frame = cap.read()
            if ret:
                name = self.recognize_faces_in_image(frame)
                if name:
                    current_time = datetime.datetime.now()
                    name_display.config(text=name)
                    process_time_display.config(text=f"{self.process_time:.2f} ms")

                    # Log to CSV
                    with open(self.attendance_log_path, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([name, current_time.strftime("%Y-%m-%d %H:%M:%S"),
                                         f"{self.process_time:.2f}"])

                    # Log to Google Sheets
                    threading.Thread(target=self.log_to_google_sheets,
                                     args=(name, current_time),
                                     daemon=True).start()

        def register_new_face():
            name = simpledialog.askstring("Register New Face", "Enter name:")
            if name:
                threading.Thread(target=self.create_new_face_directory,
                                 args=(name,),
                                 daemon=True).start()

        capture_button.config(command=capture_and_recognize)
        register_button.config(command=register_new_face)

        update_clock()
        update_frame()

        root.mainloop()
        cap.release()


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
                        model_name=model,
                        enforce_detection=False,
                        detector_backend=backend
                    )[0]['embedding']
                    known_embeddings[person_folder] = embedding
                    print(f"Loaded embedding for {person_folder}")
            except Exception as e:
                print(f"Error loading embedding for {person_folder}: {e}")

    with open(output_file, 'wb') as f:
        pickle.dump(known_embeddings, f)
    print(f"Embeddings saved to {output_file}")
    return known_embeddings


if __name__ == '__main__':
    recognition_system = LiveFaceRecognitionSystem()
    recognition_system.run_live_recognition()