import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QComboBox, QPushButton,
                             QTabWidget, QFileDialog, QProgressBar, QScrollArea,
                             QFrame, QMessageBox)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import cv2
import numpy as np
from deepface import DeepFace
from PIL import Image
import io


class FaceAnalysisApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Analysis Application")
        self.setGeometry(100, 100, 1200, 800)

        # Available models
        self.AVAILABLE_MODELS = {
            'VGG-Face': 'VGG-Face',
            'Facenet': 'Facenet',
            'Facenet512': 'Facenet512',
            'OpenFace': 'OpenFace',
            'DeepFace': 'DeepFace',
            'DeepID': 'DeepID',
            'ArcFace': 'ArcFace',
            'Dlib': 'Dlib',
            'SFace': 'SFace'
        }

        self.current_model = 'VGG-Face'
        self.setup_ui()

    def setup_ui(self):
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Create tab widget
        tabs = QTabWidget()
        layout.addWidget(tabs)

        # Create tabs
        tabs.addTab(self.create_model_tab(), "Select Model")
        tabs.addTab(self.create_analysis_tab(), "Face Analysis")
        tabs.addTab(self.create_compare_tab(), "Compare Faces")
        tabs.addTab(self.create_search_tab(), "Search Faces")

    def create_model_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Model selection
        model_label = QLabel("Choose a model:")
        self.model_combo = QComboBox()
        self.model_combo.addItems(self.AVAILABLE_MODELS.keys())
        self.model_combo.setCurrentText(self.current_model)
        self.model_combo.currentTextChanged.connect(self.update_model)

        # Info text
        info_text = QLabel(
            "Model Information:\n\n"
            "• VGG-Face: Good general-purpose model\n"
            "• Facenet: Efficient for real-time applications\n"
            "• ArcFace: High accuracy, good for verification\n"
            "• DeepFace: Balanced performance"
        )

        layout.addWidget(model_label)
        layout.addWidget(self.model_combo)
        layout.addWidget(info_text)
        layout.addStretch()

        return widget

    def create_analysis_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Image selection button
        select_btn = QPushButton("Select Image")
        select_btn.clicked.connect(self.analyze_face)
        layout.addWidget(select_btn)

        # Results display
        self.analysis_results = QLabel()
        self.analysis_image = QLabel()
        self.analysis_image.setAlignment(Qt.AlignCenter)

        layout.addWidget(self.analysis_results)
        layout.addWidget(self.analysis_image)

        return widget

    def create_compare_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Buttons layout
        btn_layout = QHBoxLayout()
        select_btn1 = QPushButton("Select First Image")
        select_btn2 = QPushButton("Select Second Image")
        compare_btn = QPushButton("Compare")

        select_btn1.clicked.connect(lambda: self.select_compare_image(1))
        select_btn2.clicked.connect(lambda: self.select_compare_image(2))
        compare_btn.clicked.connect(self.compare_faces)

        btn_layout.addWidget(select_btn1)
        btn_layout.addWidget(select_btn2)
        btn_layout.addWidget(compare_btn)

        # Images layout
        images_layout = QHBoxLayout()
        self.img1_label = QLabel()
        self.img2_label = QLabel()
        images_layout.addWidget(self.img1_label)
        images_layout.addWidget(self.img2_label)

        # Results
        self.compare_result = QLabel()
        self.similarity_progress = QProgressBar()

        layout.addLayout(btn_layout)
        layout.addLayout(images_layout)
        layout.addWidget(self.compare_result)
        layout.addWidget(self.similarity_progress)

        return widget

    def create_search_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Controls
        select_query_btn = QPushButton("Select Query Image")
        select_db_btn = QPushButton("Select Database Directory")
        search_btn = QPushButton("Search")

        select_query_btn.clicked.connect(self.select_query_image)
        select_db_btn.clicked.connect(self.select_database_dir)
        search_btn.clicked.connect(self.search_faces)

        layout.addWidget(select_query_btn)
        layout.addWidget(select_db_btn)
        layout.addWidget(search_btn)

        # Scroll area for results
        scroll = QScrollArea()
        self.search_results_widget = QWidget()
        self.search_results_layout = QVBoxLayout(self.search_results_widget)
        scroll.setWidget(self.search_results_widget)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)

        return widget

    def update_model(self, model_name):
        self.current_model = model_name

    def analyze_face(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Image files (*.jpg *.png *.jpeg)")

        if file_path:
            try:
                result = DeepFace.analyze(
                    img_path=file_path,
                    actions=['age', 'gender', 'emotion', 'race'],
                    enforce_detection=False,
                    detector_backend='retinaface'
                )[0]

                # Update results
                results_text = (
                    f"Age: {result['age']}\n"
                    f"Gender: {result['gender']}\n"
                    f"Dominant Emotion: {result['dominant_emotion']}\n"
                    f"Dominant Race: {result['dominant_race']}"
                )
                self.analysis_results.setText(results_text)

                # Display image
                pixmap = QPixmap(file_path)
                pixmap = pixmap.scaled(300, 300, Qt.KeepAspectRatio)
                self.analysis_image.setPixmap(pixmap)

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error analyzing face: {str(e)}")

    def select_compare_image(self, img_num):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Image files (*.jpg *.png *.jpeg)")

        if file_path:
            if img_num == 1:
                self.img1_path = file_path
                pixmap = QPixmap(file_path)
                pixmap = pixmap.scaled(200, 200, Qt.KeepAspectRatio)
                self.img1_label.setPixmap(pixmap)
            else:
                self.img2_path = file_path
                pixmap = QPixmap(file_path)
                pixmap = pixmap.scaled(200, 200, Qt.KeepAspectRatio)
                self.img2_label.setPixmap(pixmap)

    def compare_faces(self):
        if hasattr(self, 'img1_path') and hasattr(self, 'img2_path'):
            try:
                result = DeepFace.verify(
                    img1_path=self.img1_path,
                    img2_path=self.img2_path,
                    model_name=self.current_model,
                    enforce_detection=False,
                    detector_backend='retinaface'
                )

                verified = result['verified']
                distance = result['distance']
                similarity = max(0, min(100, (1 - distance) * 100))

                self.compare_result.setText(
                    f"Match: {'Yes' if verified else 'No'}\n"
                    f"Similarity Score: {similarity:.2f}%"
                )
                self.similarity_progress.setValue(int(similarity))

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error comparing faces: {str(e)}")
        else:
            QMessageBox.warning(self, "Warning", "Please select both images first")

    def select_query_image(self):
        self.query_image, _ = QFileDialog.getOpenFileName(
            self, "Select Query Image", "", "Image files (*.jpg *.png *.jpeg)")

    def select_database_dir(self):
        self.database_dir = QFileDialog.getExistingDirectory(
            self, "Select Database Directory")

    def search_faces(self):
        if hasattr(self, 'query_image') and hasattr(self, 'database_dir'):
            try:
                # Clear previous results
                for i in reversed(range(self.search_results_layout.count())):
                    self.search_results_layout.itemAt(i).widget().setParent(None)

                results = DeepFace.find(
                    img_path=self.query_image,
                    db_path=self.database_dir,
                    model_name=self.current_model,
                    enforce_detection=False,
                    detector_backend='retinaface'
                )

                if results[0].empty:
                    self.search_results_layout.addWidget(QLabel("No similar faces found"))
                else:
                    for _, row in results[0].iterrows():
                        result_frame = QFrame()
                        frame_layout = QHBoxLayout(result_frame)

                        # Image
                        pixmap = QPixmap(row['identity'])
                        pixmap = pixmap.scaled(100, 100, Qt.KeepAspectRatio)
                        img_label = QLabel()
                        img_label.setPixmap(pixmap)
                        frame_layout.addWidget(img_label)

                        # Similarity
                        similarity = max(0, min(100, (1 - row['distance']) * 100))
                        similarity_label = QLabel(f"Similarity: {similarity:.2f}%")
                        frame_layout.addWidget(similarity_label)

                        self.search_results_layout.addWidget(result_frame)

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error searching faces: {str(e)}")
        else:
            QMessageBox.warning(
                self, "Warning",
                "Please select both query image and database directory"
            )


def main():
    app = QApplication(sys.argv)
    window = FaceAnalysisApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()