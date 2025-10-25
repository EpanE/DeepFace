import cv2
import os
import numpy as np
from deepface import DeepFace
import time


class FaceRecognitionTest:
    def __init__(self, database_path='./employee_database'):
        self.database_path = database_path
        self.known_embeddings = {}
        self.frame_skip = 1  # Adjust this value based on Pi's performance
        self.frame_count = 0
        self.recognition_threshold = 0.3

        print("Initializing face embeddings...")
        self._load_known_faces()

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
                            model_name='VGG-Face',
                            enforce_detection=False,
                            detector_backend='retinaface'
                        )
                        if embedding and len(embedding) > 0:
                            self.known_embeddings[person_folder] = embedding[0]['embedding']
                            print(f"Loaded embedding for {person_folder}")
                except Exception as e:
                    print(f"Error loading embedding for {person_folder}: {e}")

    def _get_face_embedding(self, frame):
        """Get embedding for face in frame"""
        try:
            embedding = DeepFace.represent(
                img_path=frame,
                model_name='VGG-Face',
                enforce_detection=False,
                detector_backend='retinaface'  # Changed to opencv for better performance on Pi
            )
            if embedding and len(embedding) > 0:
                return embedding[0]['embedding']
            return None
        except Exception as e:
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
        """Detect faces in the frame using RetinaFace"""
        try:
            # Save frame temporarily for DeepFace processing
            temp_frame_path = 'temp_detect_frame.jpg'
            cv2.imwrite(temp_frame_path, frame)

            # Use DeepFace with RetinaFace backend to detect faces
            faces = DeepFace.extract_faces(
                img_path=temp_frame_path,
                detector_backend='retinaface',
                enforce_detection=False
            )

            # Clean up temporary file
            if os.path.exists(temp_frame_path):
                os.remove(temp_frame_path)

            # Convert RetinaFace output to format compatible with existing code
            face_locations = []
            for face in faces:
                facial_area = face['facial_area']
                x = facial_area['x']
                y = facial_area['y']
                w = facial_area['w']
                h = facial_area['h']
                face_locations.append((x, y, w, h))

            return face_locations
        except Exception as e:
            print(f"Face detection error: {e}")
            return []

    def recognize_face(self, frame):
        """Recognize face using pre-computed embeddings"""
        frame_embedding = self._get_face_embedding(frame)
        if frame_embedding is None:
            return []

        recognized_faces = []
        for name, known_embedding in self.known_embeddings.items():
            similarity = self._compare_embeddings(frame_embedding, known_embedding)
            if similarity > self.recognition_threshold:
                confidence = similarity * 100  # Convert to percentage
                recognized_faces.append((name, confidence))

        recognized_faces.sort(key=lambda x: x[1], reverse=True)
        return recognized_faces

    def draw_face_box(self, frame, x, y, w, h, name, confidence):
        """Draw bounding box and labels on the face"""
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Prepare text with name, confidence and FPS
        label = f"{name} ({confidence:.1f}%)"

        # Draw background rectangle for text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]

        cv2.rectangle(frame,
                      (x, y - text_size[1] - 10),
                      (x + text_size[0], y),
                      (0, 255, 0),
                      cv2.FILLED)

        # Draw text
        cv2.putText(frame,
                    label,
                    (x, y - 5),
                    font,
                    font_scale,
                    (0, 0, 0),
                    thickness)

    def run_recognition_test(self):
        """Run the face recognition test"""
        # Try to use Pi's camera if available, fallback to USB camera
        try:
            cap = cv2.VideoCapture(1)  # Try Pi camera first
            if not cap.isOpened():
                cap = cv2.VideoCapture(1)  # Fallback to USB camera
        except:
            print("Error opening camera")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print("Starting face recognition test...")
        start_time = time.time()
        frames_processed = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Calculate and display FPS
            frames_processed += 1
            if frames_processed % 30 == 0:  # Update FPS every 30 frames
                elapsed_time = time.time() - start_time
                fps = frames_processed / elapsed_time
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Detect faces in every frame
            faces = self.detect_faces(frame)

            # Process recognition every Nth frame
            if self.frame_count % self.frame_skip == 0 and len(faces) > 0:
                # Save frame temporarily
                temp_frame_path = 'temp_frame.jpg'
                cv2.imwrite(temp_frame_path, frame)

                try:
                    # Get recognition results
                    recognized_faces = self.recognize_face(temp_frame_path)

                    if recognized_faces:
                        name, confidence = recognized_faces[0]  # Get best match
                        # Draw box for each detected face
                        for (x, y, w, h) in faces:
                            self.draw_face_box(frame, x, y, w, h, name, confidence)

                    # Clean up temporary file
                    if os.path.exists(temp_frame_path):
                        os.remove(temp_frame_path)

                except Exception as e:
                    print(f"Recognition error: {e}")
            else:
                # Just draw boxes without recognition
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Show frame
            cv2.putText(frame, "Press 'q' to quit", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow('Face Recognition Test', frame)

            self.frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


def main():
    database_path = './employee_database'
    system = FaceRecognitionTest(database_path=database_path)
    system.run_recognition_test()


if __name__ == "__main__":
    main()