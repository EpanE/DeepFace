import os
import pickle
import numpy as np
import cv2
from deepface import DeepFace
from scipy.spatial.distance import cosine


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

                    # Generate embedding using Facenet512
                    embedding = DeepFace.represent(
                        img_path=image_path,
                        model_name='Facenet512',
                        enforce_detection=False,
                        detector_backend='retinaface'
                    )

                    # Store embedding if successful
                    if embedding and len(embedding) > 0:
                        known_embeddings[person_folder] = embedding[0]['embedding']
                        print(f"Loaded embedding for {person_folder}")

            except Exception as e:
                print(f"Error loading embedding for {person_folder}: {e}")

    # Save embeddings to file
    with open(output_file, 'wb') as f:
        pickle.dump(known_embeddings, f)

    print(f"Embeddings saved to {output_file}")
    return known_embeddings


class LiveFaceRecognitionSystem:
    def __init__(self, embedding_file='face_embeddings.pkl', recognition_threshold=0.3):
        """
        Initialize the Live Face Recognition System

        Args:
            embedding_file (str): Path to the pickle file containing face embeddings
            recognition_threshold (float): Cosine distance threshold for face recognition
        """
        self.known_embeddings = {}
        self.recognition_threshold = recognition_threshold
        self._last_recognition_results = []

        # Load pre-compiled embeddings
        try:
            with open(embedding_file, 'rb') as f:
                self.known_embeddings = pickle.load(f)
            print("Embeddings loaded successfully from compiled file")
        except FileNotFoundError:
            print("No compiled embeddings found. Create them first using compile_face_embeddings()")

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

            # Detect faces using DeepFace
            detection_results = DeepFace.represent(
                img_path=image,
                model_name='Facenet512',
                enforce_detection=False,
                detector_backend='retinaface'
            )

            # Compare each detected face with known embeddings
            for detection in detection_results:
                face_embedding = detection['embedding']
                region = detection.get('region', {})

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
                        'distance': lowest_distance,
                        'region': region,
                        'embedding': face_embedding
                    }
                else:
                    result = {
                        'name': 'Unknown',
                        'confidence': 0,
                        'distance': lowest_distance,
                        'region': region,
                        'embedding': face_embedding
                    }

                self._last_recognition_results.append(result)

            return self._last_recognition_results

        except Exception as e:
            print(f"Error in face recognition: {e}")
            return []

    def draw_recognition_results(self, image, results):
        """
        Draw recognition results on the image

        Args:
            image (numpy.ndarray): Image to annotate
            results (list): List of recognition results

        Returns:
            numpy.ndarray: Annotated image
        """
        annotated_image = image.copy()

        for result in results:
            # Skip if no region detected
            if not result['region']:
                continue

            # Extract region coordinates
            x = result['region']['x']
            y = result['region']['y']
            w = result['region']['w']
            h = result['region']['h']

            # Choose color based on recognition
            color = (0, 255, 0) if result['name'] != 'Unknown' else (0, 0, 255)

            # Draw rectangle around the face
            cv2.rectangle(annotated_image, (x, y), (x + w, y + h), color, 2)

            # Prepare text
            label = f"{result['name']} (Conf: {result['confidence']:.2f})"

            # Calculate text size for background
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
            )

            # Draw text background
            cv2.rectangle(
                annotated_image,
                (x, y - text_height - 10),
                (x + text_width, y),
                color,
                -1
            )

            # Put text with name and confidence
            cv2.putText(
                annotated_image,
                label,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1
            )

        return annotated_image

    def run_live_recognition(self, video_source=1):
        """
        Run live recognition system with on-demand face detection

        Args:
            video_source (int): Camera index (default 0 for primary webcam)
        """
        # Open video capture
        cap = cv2.VideoCapture(video_source)

        # Window to display camera feed
        cv2.namedWindow('Live Camera', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Live Camera', 640, 480)

        while True:
            # Read a frame from the video capture
            ret, frame = cap.read()

            if not ret:
                print("Failed to grab frame")
                break

            # Display the current frame
            cv2.imshow('Live Camera', frame)

            # Wait for key press
            key = cv2.waitKey(1) & 0xFF

            # 'p' key pressed - perform face recognition
            if key == ord('p'):
                try:
                    # Perform face recognition on the current frame
                    results = self.recognize_faces_in_image(frame)

                    if results:
                        # Draw recognition results
                        annotated_frame = self.draw_recognition_results(frame, results)

                        # Display detailed information about detected faces
                        cv2.namedWindow('Face Recognition Results', cv2.WINDOW_NORMAL)
                        cv2.resizeWindow('Face Recognition Results', 640, 480)
                        cv2.imshow('Face Recognition Results', annotated_frame)

                        # Print detailed recognition information
                        print("\nRecognition Results:")
                        for result in results:
                            print(f"Name: {result['name']}")
                            print(f"Confidence: {result['confidence']:.2f}")
                            print(f"Distance: {result['distance']:.4f}")
                            print("---")

                    else:
                        print("No faces detected.")

                except Exception as e:
                    print(f"Error processing frame: {e}")

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

    # Initialize live recognition system
    recognition_system = LiveFaceRecognitionSystem()

    # Run live recognition
    recognition_system.run_live_recognition()