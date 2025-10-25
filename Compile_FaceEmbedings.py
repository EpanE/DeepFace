import os
import pickle
from deepface import DeepFace


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

                    # Generate embedding
                    embedding = DeepFace.represent(
                        img_path=image_path,
                        model_name='Facenet',
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


# Modify FaceRecognitionSystem to load from compiled file
class FaceRecognitionSystem:
    def __init__(self, embedding_file='face_embeddings.pkl'):
        self.known_embeddings = {}
        self.recognition_threshold = 0.3
        self._last_recognition_results = []

        # Load pre-compiled embeddings
        try:
            with open(embedding_file, 'rb') as f:
                self.known_embeddings = pickle.load(f)
            print("Embeddings loaded successfully from compiled file")
        except FileNotFoundError:
            print("No compiled embeddings found. Create them first using compile_face_embeddings()")


# Usage example
if __name__ == '__main__':
    # First, compile embeddings
    compile_face_embeddings()

    # Then initialize recognition system with compiled file
    recognition_system = FaceRecognitionSystem()