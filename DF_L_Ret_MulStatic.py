import cv2
import os
import numpy as np
from deepface import DeepFace
import mediapipe as mp
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import threading


class FaceRecognitionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Multi-Image Face Recognition System")
        self.root.geometry("1400x900")

        # Make the root window responsive
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Initialize face recognition system
        self.recognition_system = FaceRecognitionSystem()

        # Create GUI elements
        self.create_gui()

    def create_gui(self):
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")

        # Configure main frame grid weights
        main_frame.columnconfigure(1, weight=3)  # Right frame takes more space
        main_frame.columnconfigure(0, weight=1)  # Left frame takes less space
        main_frame.rowconfigure(0, weight=1)

        # Create left frame for controls
        left_frame = ttk.Frame(main_frame, padding="5")
        left_frame.grid(row=0, column=0, sticky="nsew")
        left_frame.columnconfigure(0, weight=1)

        # Database section
        ttk.Label(left_frame, text="Database Management", font=('Arial', 12, 'bold')).grid(row=0, column=0, pady=5,
                                                                                           sticky="w")

        self.database_path = tk.StringVar(value="./employee_database")
        ttk.Label(left_frame, text="Database Path:").grid(row=1, column=0, pady=2, sticky="w")
        ttk.Entry(left_frame, textvariable=self.database_path).grid(row=2, column=0, pady=2, sticky="ew")
        ttk.Button(left_frame, text="Browse Database", command=self.browse_database).grid(row=3, column=0, pady=5,
                                                                                          sticky="ew")
        ttk.Button(left_frame, text="Load Database", command=self.load_database).grid(row=4, column=0, pady=5,
                                                                                      sticky="ew")

        # Image selection section
        ttk.Separator(left_frame, orient='horizontal').grid(row=5, column=0, pady=10, sticky='ew')
        ttk.Label(left_frame, text="Image Processing", font=('Arial', 12, 'bold')).grid(row=6, column=0, pady=5,
                                                                                        sticky="w")

        ttk.Button(left_frame, text="Select Images", command=self.browse_images).grid(row=7, column=0, pady=5,
                                                                                      sticky="ew")
        ttk.Button(left_frame, text="Process Images", command=self.process_images).grid(row=8, column=0, pady=5,
                                                                                        sticky="ew")

        # Processing progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(left_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=9, column=0, pady=5, sticky="ew")

        # Status section
        ttk.Separator(left_frame, orient='horizontal').grid(row=10, column=0, pady=10, sticky='ew')
        ttk.Label(left_frame, text="Status", font=('Arial', 12, 'bold')).grid(row=11, column=0, pady=5, sticky="w")

        # Status text with scrollbar
        self.status_text = scrolledtext.ScrolledText(left_frame, wrap=tk.WORD, height=10)
        self.status_text.grid(row=12, column=0, pady=5, sticky="nsew")
        left_frame.rowconfigure(12, weight=1)

        # Create right frame for image display
        right_frame = ttk.Frame(main_frame, padding="5")
        right_frame.grid(row=0, column=1, sticky="nsew")

        # Configure right frame grid weights
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(1, weight=1)

        # Processed images display with scrollbar
        ttk.Label(right_frame, text="Processed Images", font=('Arial', 12, 'bold')).grid(row=0, column=0, pady=5)

        # Create a canvas with scrollbar for processed images
        self.images_canvas = tk.Canvas(right_frame)
        self.images_canvas.grid(row=1, column=0, sticky="nsew")

        # Scrollbar for canvas
        scrollbar = ttk.Scrollbar(right_frame, orient="vertical", command=self.images_canvas.yview)
        scrollbar.grid(row=1, column=1, sticky="ns")
        self.images_canvas.configure(yscrollcommand=scrollbar.set)

        # Frame inside canvas to hold images
        self.images_frame = ttk.Frame(self.images_canvas)
        self.images_canvas.create_window((0, 0), window=self.images_frame, anchor="nw")

        # Initialize variables
        self.input_image_paths = []
        self.processed_images = []
        self.photo_images = []

        # Bind events
        self.images_frame.bind("<Configure>", self.on_frame_configure)
        self.root.bind("<Configure>", self.on_window_resize)

    def on_frame_configure(self, event):
        """Reset the scroll region to encompass the inner frame"""
        self.images_canvas.configure(scrollregion=self.images_canvas.bbox("all"))

    def on_window_resize(self, event):
        """Handle window resize events"""
        if event.widget == self.root:
            self.on_frame_configure(event)

    def update_status(self, message):
        """Update status text with new message"""
        self.status_text.insert(tk.END, f"{message}\n")
        self.status_text.see(tk.END)
        self.root.update_idletasks()

    def browse_database(self):
        """Browse and select database directory"""
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.database_path.set(folder_path)
            self.update_status(f"Database path set to: {folder_path}")

    def load_database(self):
        """Load face recognition database"""
        try:
            self.recognition_system = FaceRecognitionSystem(self.database_path.get())
            self.update_status("Database loaded successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load database: {str(e)}")
            self.update_status(f"Error loading database: {str(e)}")

    def browse_images(self):
        """Browse and select multiple image files"""
        file_paths = filedialog.askopenfilenames(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff")]
        )
        if file_paths:
            self.input_image_paths = list(file_paths)
            # Clear previous images
            for widget in self.images_frame.winfo_children():
                widget.destroy()
            self.photo_images.clear()

            # Display selected images
            for path in self.input_image_paths:
                self.display_input_image(path)

            self.update_status(f"Selected {len(file_paths)} images")

    def display_input_image(self, image_path):
        """Display input image in the images frame"""
        try:
            # Open and resize image
            img = Image.open(image_path)
            img.thumbnail((300, 300))  # Limit image size
            photo = ImageTk.PhotoImage(img)

            # Create label for image
            img_frame = ttk.Frame(self.images_frame)
            img_frame.pack(pady=5)

            img_label = ttk.Label(img_frame, image=photo)
            img_label.image = photo  # Keep a reference
            img_label.pack(side=tk.LEFT)

            # Add filename label
            filename = os.path.basename(image_path)
            filename_label = ttk.Label(img_frame, text=filename)
            filename_label.pack(side=tk.LEFT, padx=10)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to display image: {str(e)}")

    def process_images(self):
        """Process multiple images in a separate thread"""
        if not self.input_image_paths:
            messagebox.showwarning("Warning", "Please select images first")
            return

        # Clear previous processed images
        for widget in self.images_frame.winfo_children():
            widget.destroy()
        self.photo_images.clear()
        self.processed_images.clear()

        # Start processing in a separate thread
        threading.Thread(target=self._process_images_thread, daemon=True).start()

    def _process_images_thread(self):
        """Threading method to process images with detailed results"""
        try:
            # Clear previous status
            self.root.after(0, self.status_text.delete, '1.0', tk.END)

            # Process images
            total_images = len(self.input_image_paths)
            processed_count = 0

            # Prepare results string
            results_summary = "Image Processing Results:\n"
            results_summary += "=" * 50 + "\n\n"

            for image_path in self.input_image_paths:
                # Process single image
                processed_image = self.recognition_system.process_image(image_path)
                self.processed_images.append(processed_image)

                # Get recognition results
                filename = os.path.basename(image_path)
                recognition_results = self.recognition_system.get_last_recognition_results()

                # Prepare result string for this image
                results_summary += f"Image: {filename}\n"
                if recognition_results:
                    for name, confidence in recognition_results:
                        results_summary += f"  - Recognized: {name} (Confidence: {confidence:.2f}%)\n"
                else:
                    results_summary += "  - No faces recognized\n"
                results_summary += "\n"

                # Update progress
                processed_count += 1
                progress_percent = (processed_count / total_images) * 100
                self.progress_var.set(progress_percent)

                # Display processed image (thread-safe update)
                self.root.after(0, self.display_processed_image, image_path, processed_image)

            # Update status text with results
            self.root.after(0, self.update_status, results_summary)

            # Final update
            self.root.after(0, self.finalize_processing, total_images)

        except Exception as e:
            self.root.after(0, messagebox.showerror, "Processing Error", str(e))

    def display_processed_image(self, original_path, processed_image):
        """Display processed image in the images frame"""
        try:
            # Convert OpenCV image to PIL
            img = Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
            img.thumbnail((600, 600))  # Limit image size
            photo = ImageTk.PhotoImage(img)

            # Create frame for processed image
            img_frame = ttk.Frame(self.images_frame)
            img_frame.pack(pady=5)

            img_label = ttk.Label(img_frame, image=photo)
            img_label.image = photo  # Keep a reference
            img_label.pack(side=tk.LEFT)

            # Add filename label
            filename = os.path.basename(original_path)
            filename_label = ttk.Label(img_frame, text=filename)
            filename_label.pack(side=tk.LEFT, padx=10)

            self.photo_images.append(photo)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to display processed image: {str(e)}")

    def finalize_processing(self, total_images):
        """Finalize processing and update UI"""
        self.progress_var.set(0)
        self.update_status(f"Processed {total_images} images successfully")
        messagebox.showinfo("Processing Complete", f"Processed {total_images} images")


# The rest of the code (FaceRecognitionSystem class and main function) remains the same as in the original implementation
# This means you would include the entire FaceRecognitionSystem class and the main() function from the original code
class FaceRecognitionSystem:
    def __init__(self, database_path='./employee_database'):
        self.database_path = database_path
        self.known_embeddings = {}
        self.recognition_threshold = 0.3
        self._last_recognition_results = []  # Store recognition results for the last processed image

        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5
        )

        print("Initializing face embeddings...")
        self._load_known_faces()

    def get_last_recognition_results(self):
        """Return the recognition results from the last processed image"""
        return self._last_recognition_results

    def _load_known_faces(self):
        """Pre-load and cache face embeddings for known faces"""
        for person_folder in os.listdir(self.database_path):
            folder_path = os.path.join(self.database_path, person_folder)
            if os.path.isdir(folder_path):
                try:
                    image_files = [f for f in os.listdir(folder_path)
                                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    if image_files:
                        image_path = os.path.join(folder_path, image_files[0])
                        embedding = DeepFace.represent(
                            img_path=image_path,
                            model_name='Facenet',
                            enforce_detection=False,
                            detector_backend='retinaface'
                        )
                        if embedding and len(embedding) > 0:
                            self.known_embeddings[person_folder] = embedding[0]['embedding']
                            print(f"Loaded embedding for {person_folder}")
                except Exception as e:
                    print(f"Error loading embedding for {person_folder}: {e}")

    def _get_face_embedding(self, image_path):
        """Get embedding for face in image"""
        try:
            embedding = DeepFace.represent(
                img_path=image_path,
                model_name='Facenet',
                enforce_detection=False,
                detector_backend='retinaface'
            )
            if embedding and len(embedding) > 0:
                return embedding[0]['embedding']
            return None
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return None

    def _compare_embeddings(self, embedding1, embedding2):
        """Compare two face embeddings using cosine similarity"""
        try:
            if embedding1 is None or embedding2 is None:
                return 0

            vec1 = np.array(embedding1).flatten()
            vec2 = np.array(embedding2).flatten()

            similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            return similarity
        except Exception as e:
            return 0

    def detect_faces(self, frame):
        """Detect faces in the frame using MediaPipe"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(frame_rgb)

        faces = []
        if results.detections:
            frame_height, frame_width, _ = frame.shape
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * frame_width)
                y = int(bbox.ymin * frame_height)
                w = int(bbox.width * frame_width)
                h = int(bbox.height * frame_height)

                x = max(0, x)
                y = max(0, y)
                w = min(w, frame_width - x)
                h = min(h, frame_height - y)

                faces.append((x, y, w, h, detection.score[0]))

        return faces

    def recognize_face(self, image_path):
        """Recognize face in the image"""
        frame_embedding = self._get_face_embedding(image_path)
        if frame_embedding is None:
            self._last_recognition_results = []
            return []

        recognized_faces = []
        for name, known_embedding in self.known_embeddings.items():
            similarity = self._compare_embeddings(frame_embedding, known_embedding)
            if similarity > self.recognition_threshold:
                confidence = similarity * 100
                recognized_faces.append((name, confidence))

        recognized_faces.sort(key=lambda x: x[1], reverse=True)

        # Store results for retrieval
        self._last_recognition_results = recognized_faces
        return recognized_faces

    def draw_face_box(self, frame, x, y, w, h, name, confidence, detection_score):
        """Draw bounding box and labels on the face"""
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        label = f"{name} ({confidence:.1f}%) [Det: {detection_score:.2f}]"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]

        cv2.rectangle(frame,
                      (x, y - text_size[1] - 10),
                      (x + text_size[0], y),
                      (0, 255, 0),
                      cv2.FILLED)

        cv2.putText(frame,
                    label,
                    (x, y - 5),
                    font,
                    font_scale,
                    (0, 0, 0),
                    thickness)

    def process_image(self, input_image_path):
        """Process a single image and perform face recognition"""
        frame = cv2.imread(input_image_path)
        if frame is None:
            raise Exception(f"Could not read image {input_image_path}")

        faces = self.detect_faces(frame)
        if not faces:
            print("No faces detected in the image")
            return frame

        recognized_faces = self.recognize_face(input_image_path)

        if recognized_faces:
            name, confidence = recognized_faces[0]
            for (x, y, w, h, detection_score) in faces:
                self.draw_face_box(frame, x, y, w, h, name, confidence, detection_score)
        else:
            for (x, y, w, h, detection_score) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, f"Unknown [Det: {detection_score:.2f}]",
                            (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 3)

        return frame




def main():
    root = tk.Tk()
    app = FaceRecognitionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()