import cv2
import os
import pandas as pd
from datetime import datetime
from deepface import DeepFace


class AttendanceSystem:
    def __init__(self, models=None, database_path='./dataset'):
        if models is None:
            models = ['VGG-Face', 'Facenet', 'DeepID']
        self.models = models
        self.database_path = database_path
        self.attendance_log = pd.DataFrame(columns=['Name', 'Timestamp'])

        # Validate database path exists
        if not os.path.exists(database_path):
            raise ValueError(f"Database path {database_path} does not exist")

    def recognize_face(self, frame_path):
        results = []
        for model in self.models:
            try:
                # Find similar faces in database
                result = DeepFace.find(
                    img_path=frame_path,
                    db_path=self.database_path,
                    model_name=model,
                    enforce_detection=False,
                    detector_backend='retinaface'
                )

                # Extract unique identities
                identities = result[0]['identity'].tolist()
                results.extend(identities)
            except Exception as e:
                print(f"Error with {model}: {e}")

        # Remove duplicates and return
        return list(set(results))

    def mark_attendance(self, recognized_faces):
        current_time = datetime.now()
        for face in recognized_faces:
            # Check if already marked today
            existing_entries = self.attendance_log[
                (self.attendance_log['Name'] == face) &
                (pd.to_datetime(self.attendance_log['Timestamp']).dt.date == current_time.date())
                ]

            if existing_entries.empty:
                new_entry = pd.DataFrame({
                    'Name': [face],
                    'Timestamp': [current_time]
                })
                self.attendance_log = pd.concat([self.attendance_log, new_entry], ignore_index=True)
                print(f"Attendance marked for {face}")

    def run_attendance_system(self, confidence_threshold=0.5):
        cap = cv2.VideoCapture(0)  # Open webcam

        # Set lower resolution for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Temporary save frame for processing
            temp_frame_path = 'temp_frame.jpg'
            cv2.imwrite(temp_frame_path, frame)

            try:
                # Recognize faces
                recognized_faces = self.recognize_face(temp_frame_path)

                if recognized_faces:
                    self.mark_attendance(recognized_faces)

                # Draw rectangles around detected faces
                for face in recognized_faces:
                    cv2.putText(frame, face, (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 0), 2)

            except Exception as e:
                print(f"Face recognition error: {e}")

            # Display frame
            cv2.imshow('Attendance System', frame)

            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Cleanup
        cap.release()
        cv2.destroyAllWindows()

        # Save attendance log
        log_path = f'attendance_log_{datetime.now().strftime("%Y%m%d")}.csv'
        self.attendance_log.to_csv(log_path, index=False)
        print(f"Attendance log saved to {log_path}")


def main():
    # Ensure database path is set correctly
    database_path = './dataset'

    # Initialize and run attendance system
    system = AttendanceSystem(database_path=database_path)
    system.run_attendance_system()


if __name__ == "__main__":
    main()