from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt
import os


def analyze_face(image_path):
    """
    Analyze facial attributes in an image using DeepFace

    Parameters:
    image_path (str): Path to the image file

    Returns:
    dict: Dictionary containing facial analysis results
    """
    try:
        # Analyze facial attributes
        result = DeepFace.analyze(
            img_path=image_path,
            actions=['age', 'gender', 'emotion', 'race'],
            enforce_detection=False
        )
        return result[0]  # DeepFace returns a list with one dictionary
    except Exception as e:
        print(f"Error analyzing face: {str(e)}")
        return None


def verify_faces(img1_path, img2_path):
    """
    Verify if two face images belong to the same person

    Parameters:
    img1_path (str): Path to the first image
    img2_path (str): Path to the second image

    Returns:
    tuple: (boolean verification result, float verification score)
    """
    try:
        result = DeepFace.verify(
            img1_path=img1_path,
            img2_path=img2_path,
            enforce_detection=False
        )
        return result['verified'], result['distance']
    except Exception as e:
        print(f"Error verifying faces: {str(e)}")
        return None, None


def find_similar_faces(image_path, database_path):
    """
    Find similar faces in a database

    Parameters:
    image_path (str): Path to the query image
    database_path (str): Path to the directory containing database images

    Returns:
    list: List of paths to similar face images
    """
    try:
        results = DeepFace.find(
            img_path=image_path,
            db_path=database_path,
            enforce_detection=False
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
    # Read and display image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.axis('off')

    # Display analysis results
    if analysis_result:
        print("\nFacial Analysis Results:")
        print(f"Age: {analysis_result['age']}")
        print(f"Gender: {analysis_result['gender']}")
        print(f"Dominant Emotion: {analysis_result['dominant_emotion']}")
        print(f"Dominant Race: {analysis_result['dominant_race']}")

    plt.show()


# Example usage
if __name__ == "__main__":
    # Replace these paths with your actual image paths
    image_path = "photo_2023-09-13_16-15-13.jpg"
    database_path = "dataset"

    # Analyze single face
    print("Analyzing face...")
    result = analyze_face(image_path)
    if result:
        display_results(image_path, result)

    # Compare two faces
    image2_path = "GAMBAR_GUE-removebg-preview.png"
    print("\nVerifying faces...")
    verified, score = verify_faces(image_path, image2_path)
    if verified is not None:
        print(f"Faces match: {verified}")
        print(f"Similarity score: {score}")

    # Find similar faces in database
    print("\nFinding similar faces...")
    similar_faces = find_similar_faces(image_path, database_path)
    if similar_faces:
        print("Similar faces found in:")
        for face_path in similar_faces:
            print(face_path)