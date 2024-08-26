import sys
import requests
import numpy as np
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QFileDialog, QInputDialog, QGridLayout, QLabel
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, QSize, Qt

class VideoApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Video Detection')
        self.setGeometry(100, 100, 1200, 800)
        self.video_source = None
        self.capture = None
        self.stream_generator = None
        self.show_effect = False
        self.effect_bg = False
        self.effect_detection = False

        # Initialize Background Subtractor
        self.fgbg = cv2.createBackgroundSubtractorMOG2()

        # Create central widget and main layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QGridLayout()
        self.central_widget.setLayout(self.main_layout)

        # Create sidebar for navigation
        self.sidebar_widget = QWidget()
        self.sidebar_layout = QVBoxLayout()
        self.sidebar_layout.setContentsMargins(10, 10, 10, 10)
        self.sidebar_layout.setSpacing(10)
        self.sidebar_widget.setLayout(self.sidebar_layout)
        self.main_layout.addWidget(self.sidebar_widget, 0, 0, 4, 1)

        # Create buttons for sidebar
        self.create_sidebar_buttons()

        # Create video and effect areas
        self.video_label = QLabel()
        self.effect_bg_label = QLabel()
        self.effect_detection_label = QLabel()
        self.effect_bg_label.setStyleSheet("border: 1px solid green;")  # Border to differentiate background subtraction area
        self.effect_detection_label.setStyleSheet("border: 1px solid blue;")  # Border to differentiate object detection area

        # Add video and effect labels to the layout
        self.main_layout.addWidget(self.video_label, 0, 1, 2, 2)
        self.main_layout.addWidget(self.effect_bg_label, 2, 1)
        self.main_layout.addWidget(self.effect_detection_label, 2, 2)

        # Initialize timer for video update
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def create_sidebar_buttons(self):
        # Phone Camera Button
        self.phone_btn = QPushButton('Phone Camera')
        self.phone_btn.clicked.connect(self.connect_phone)
        self.phone_btn.setStyleSheet("background-color: #3498db; color: white; border: none; padding: 10px; border-radius: 5px;")
        self.sidebar_layout.addWidget(self.phone_btn)

        # Webcam Button
        self.webcam_btn = QPushButton('Webcam')
        self.webcam_btn.clicked.connect(self.connect_webcam)
        self.webcam_btn.setStyleSheet("background-color: #3498db; color: white; border: none; padding: 10px; border-radius: 5px;")
        self.sidebar_layout.addWidget(self.webcam_btn)

        # Video File Button
        self.video_btn = QPushButton('Video File')
        self.video_btn.clicked.connect(self.load_video)
        self.video_btn.setStyleSheet("background-color: #3498db; color: white; border: none; padding: 10px; border-radius: 5px;")
        self.sidebar_layout.addWidget(self.video_btn)

        # Effect Buttons
        self.bg_sub_btn = QPushButton('Background Subtraction')
        self.bg_sub_btn.clicked.connect(self.toggle_bg_subtraction)
        self.bg_sub_btn.setStyleSheet("background-color: #2ecc71; color: white; border: none; padding: 10px; border-radius: 5px;")
        self.sidebar_layout.addWidget(self.bg_sub_btn)
        
        self.obj_det_btn = QPushButton('Object Detection')
        self.obj_det_btn.clicked.connect(self.toggle_object_detection)
        self.obj_det_btn.setStyleSheet("background-color: #e67e22; color: white; border: none; padding: 10px; border-radius: 5px;")
        self.sidebar_layout.addWidget(self.obj_det_btn)
        
        self.stop_btn = QPushButton('Stop')
        self.stop_btn.clicked.connect(self.stop_video)
        self.stop_btn.setStyleSheet("background-color: #e74c3c; color: white; border: none; padding: 10px; border-radius: 5px;")
        self.sidebar_layout.addWidget(self.stop_btn)

        # Set sidebar properties
        self.sidebar_widget.setFixedWidth(200)
        self.sidebar_widget.setStyleSheet("background-color: #2c3e50;")

    def connect_phone(self):
        url, ok = QInputDialog.getText(self, 'Phone Camera URL', 'Enter the URL of the DroidCam:')
        if ok and url:
            self.video_source = url
            self.capture = None
            self.stream_generator = self.get_stream(url)
            self.effect_bg_label.clear()
            self.effect_detection_label.clear()
        else:
            print("Please enter a valid URL.")

    def connect_webcam(self):
        self.video_source = 0  # Default webcam
        self.capture = cv2.VideoCapture(self.video_source)
        self.stream_generator = None
        self.effect_bg_label.clear()
        self.effect_detection_label.clear()

    def load_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Select Video File', '', 'Video Files (*.mp4 *.avi)')
        if file_path:
            self.video_source = file_path
            self.capture = cv2.VideoCapture(self.video_source)
            self.stream_generator = None
            self.effect_bg_label.clear()
            self.effect_detection_label.clear()

    def toggle_bg_subtraction(self):
        self.effect_bg = not self.effect_bg
        if not self.effect_bg and not self.effect_detection:
            self.show_effect = False
            self.effect_bg_label.clear()
        else:
            self.show_effect = True
            self.effect_bg_label.setText('Background Subtraction Effect')
        
    def toggle_object_detection(self):
        self.effect_detection = not self.effect_detection
        if not self.effect_detection and not self.effect_bg:
            self.show_effect = False
            self.effect_detection_label.clear()
        else:
            self.show_effect = True
            self.effect_detection_label.setText('Object Detection Effect')

    def stop_video(self):
        if self.capture:
            self.capture.release()
        self.video_label.clear()
        self.stream_generator = None
        self.effect_bg_label.clear()
        self.effect_detection_label.clear()

    def get_stream(self, url):
        def stream():
            stream = requests.get(url, stream=True, timeout=10)
            bytes_data = b''
            while True:
                for chunk in stream.iter_content(chunk_size=1024):
                    bytes_data += chunk
                    a = bytes_data.find(b'\xff\xd8')  # Início do JPEG
                    b = bytes_data.find(b'\xff\xd9')  # Fim do JPEG
                    if a != -1 and b != -1:
                        jpg = bytes_data[a:b+2]
                        bytes_data = bytes_data[b+2:]
                        img_array = np.frombuffer(jpg, dtype=np.uint8)
                        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                        if img is not None:
                            yield img
        return stream()

    def apply_background_subtraction_effect(self, frame):
        fgmask = self.fgbg.apply(frame)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        return fgmask

    def apply_object_detection_effect(self, frame):
        fgmask = self.fgbg.apply(frame)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 500:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        return frame

    def update_frame(self):
        if self.stream_generator:
            try:
                frame = next(self.stream_generator, None)
                if frame is None:
                    return
            except StopIteration:
                self.stream_generator = None
                return
        elif self.capture:
            ret, frame = self.capture.read()
            if not ret:
                return
        else:
            return

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qimg_frame = QImage(frame_rgb.data, frame_rgb.shape[1], frame_rgb.shape[0], QImage.Format_RGB888)
        pixmap_frame = QPixmap.fromImage(qimg_frame)
        self.video_label.setPixmap(pixmap_frame)

        if self.show_effect:
            if self.effect_bg:
                fgmask = self.apply_background_subtraction_effect(frame)
                fgmask_rgb = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2RGB)
                qimg_fgmask = QImage(fgmask_rgb.data, fgmask_rgb.shape[1], fgmask_rgb.shape[0], QImage.Format_RGB888)
                pixmap_fgmask = QPixmap.fromImage(qimg_fgmask)
                self.effect_bg_label.setPixmap(pixmap_fgmask)

            if self.effect_detection:
                detected_frame = self.apply_object_detection_effect(frame)
                detected_frame_rgb = cv2.cvtColor(detected_frame, cv2.COLOR_BGR2RGB)
                qimg_detected = QImage(detected_frame_rgb.data, detected_frame_rgb.shape[1], detected_frame_rgb.shape[0], QImage.Format_RGB888)
                pixmap_detected = QPixmap.fromImage(qimg_detected)
                self.effect_detection_label.setPixmap(pixmap_detected)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = VideoApp()
    window.show()
    sys.exit(app.exec_())
