from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt
import os

# Available models in DeepFace
AVAILABLE_MODELS = {
    '1': 'VGG-Face',
    '2': 'Facenet',
    '3': 'Facenet512',
    '4': 'OpenFace',
    '5': 'DeepFace',
    '6': 'DeepID',
    '7': 'ArcFace',
    '8': 'Dlib',
    '9': 'SFace'
}


def select_model():
    """
    Allows user to select the face recognition model

    Returns:
    str: Selected model name
    """
    print("\nAvailable Face Recognition Models:")
    for key, model in AVAILABLE_MODELS.items():
        print(f"{key}: {model}")

    while True:
        choice = input("\nSelect a model (enter number): ")
        if choice in AVAILABLE_MODELS:
            return AVAILABLE_MODELS[choice]
        print("Invalid choice. Please try again.")


def analyze_face(image_path, model_name='VGG-Face'):
    """
    Analyze facial attributes in an image using DeepFace

    Parameters:
    image_path (str): Path to the image file
    model_name (str): Name of the face recognition model to use

    Returns:
    dict: Dictionary containing facial analysis results
    """
    try:
        result = DeepFace.analyze(
            img_path=image_path,
            actions=['age', 'gender', 'emotion', 'race'],
            enforce_detection=False,
            detector_backend='retinaface',
            model_name=model_name
        )
        return result[0]
    except Exception as e:
        print(f"Error analyzing face: {str(e)}")
        return None


def verify_faces(img1_path, img2_path, model_name='VGG-Face'):
    """
    Verify if two face images belong to the same person

    Parameters:
    img1_path (str): Path to the first image
    img2_path (str): Path to the second image
    model_name (str): Name of the face recognition model to use

    Returns:
    tuple: (boolean verification result, float verification score)
    """
    try:
        result = DeepFace.verify(
            img1_path=img1_path,
            img2_path=img2_path,
            model_name=model_name,
            enforce_detection=False,
            detector_backend='retinaface'
        )
        return result['verified'], result['distance']
    except Exception as e:
        print(f"Error verifying faces: {str(e)}")
        return None, None


def find_similar_faces(image_path, database_path, model_name='VGG-Face'):
    """
    Find similar faces in a database

    Parameters:
    image_path (str): Path to the query image
    database_path (str): Path to the directory containing database images
    model_name (str): Name of the face recognition model to use

    Returns:
    list: List of paths to similar face images
    """
    try:
        results = DeepFace.find(
            img_path=image_path,
            db_path=database_path,
            model_name=model_name,
            enforce_detection=False,
            detector_backend='retinaface'
        )
        return results[0]['identity'].tolist()
    except Exception as e:
        print(f"Error finding similar faces: {str(e)}")
        return []


def display_results(image_path, analysis_result):
    """
    Display the image and analysis results

    Parameters:
    image_path (str): Path to the image file
    analysis_result (dict): Results from facial analysis
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.axis('off')

    if analysis_result:
        print("\nFacial Analysis Results:")
        print(f"Age: {analysis_result['age']}")
        print(f"Gender: {analysis_result['gender']}")
        print(f"Dominant Emotion: {analysis_result['dominant_emotion']}")
        print(f"Dominant Race: {analysis_result['dominant_race']}")

    plt.show()


def main():
    # Replace these paths with your actual image paths
    image_path = "GAMBAR_GUE-removebg-preview.png"
    database_path = "dataset"

    # Select face recognition model
    model_name = select_model()
    print(f"\nUsing model: {model_name}")

    # Analyze single face
    print("\nAnalyzing face...")
    result = analyze_face(image_path, model_name)
    if result:
        display_results(image_path, result)

    # Compare two faces
    image2_path = "photo_2023-09-13_16-15-13.jpg"
    print("\nVerifying faces...")
    verified, score = verify_faces(image_path, image2_path, model_name)
    if verified is not None:
        print(f"Faces match: {verified}")
        print(f"Similarity score: {score}")

    # Find similar faces in database
    print("\nFinding similar faces...")
    similar_faces = find_similar_faces(image_path, database_path, model_name)
    if similar_faces:
        print("Similar faces found in:")
        for face_path in similar_faces:
            print(face_path)


if __name__ == "__main__":
    main()