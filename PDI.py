import sys
import cv2
import numpy as np
import requests
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget,
    QFileDialog, QInputDialog, QLabel, QHBoxLayout, QFrame, QMessageBox
)
from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon, QColor
from PyQt5.QtCore import QTimer, Qt, QSize

class AplicativoDeVideo(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Sistema de Monitoramento Inteligente')
        self.setGeometry(100, 100, 1200, 800)
        self.video_source = None
        self.capture = None
        self.stream_generator = None
        self.show_effect = False
        self.effect_bg = False
        self.effect_detection = False

        # Carregar o modelo YOLOv4-tiny
        self.net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
        self.model = cv2.dnn_DetectionModel(self.net)
        self.model.setInputParams(size=(416, 416), scale=1/255)

        # Carregar as classes COCO
        with open("coco.names", "r") as f:
            self.class_names = [cname.strip() for cname in f.readlines()]

        # Cores para as classes
        self.COLORS = np.random.uniform(0, 255, size=(len(self.class_names), 3))

        # Inicializar o Subtrator de Fundo
        self.fgbg = cv2.createBackgroundSubtractorMOG2()

        # Configurar interface
        self.init_ui()

        # Inicializar o temporizador para atualização do vídeo
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def init_ui(self):
        # Fonte padrão
        self.font = QFont("Segoe UI", 10)

        # Layout principal
        main_layout = QHBoxLayout()
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Barra lateral
        sidebar = QFrame()
        sidebar.setFixedWidth(200)
        sidebar.setStyleSheet("background-color: #2c3e50;")
        sidebar_layout = QVBoxLayout()
        sidebar_layout.setContentsMargins(20, 20, 20, 20)
        sidebar_layout.setSpacing(15)
        sidebar.setLayout(sidebar_layout)

        # Botões da barra lateral
        self.btn_phone_camera = self.create_button(
            "Câmera do Telefone", "icons/phone.png", self.connect_phone_camera
        )
        self.btn_webcam = self.create_button(
            "Webcam", "icons/webcam.png", self.connect_webcam
        )
        self.btn_video_file = self.create_button(
            "Arquivo de Vídeo", "icons/video_file.png", self.load_video_file
        )
        self.btn_bg_subtraction = self.create_button(
            "Subtração de Fundo", "icons/bg_subtraction.png", self.toggle_bg_subtraction, checkable=True
        )
        self.btn_object_detection = self.create_button(
            "Detecção de Objetos", "icons/object_detection.png", self.toggle_object_detection, checkable=True
        )
        self.btn_stop = self.create_button(
            "Parar", "icons/stop.png", self.stop_video
        )

        # Adicionar botões ao layout da barra lateral
        sidebar_layout.addWidget(self.btn_phone_camera)
        sidebar_layout.addWidget(self.btn_webcam)
        sidebar_layout.addWidget(self.btn_video_file)
        sidebar_layout.addStretch()
        sidebar_layout.addWidget(self.btn_bg_subtraction)
        sidebar_layout.addWidget(self.btn_object_detection)
        sidebar_layout.addStretch()
        sidebar_layout.addWidget(self.btn_stop)

        # Área de exibição de vídeo
        video_frame = QFrame()
        video_frame.setStyleSheet("background-color: #ecf0f1; border-radius: 10px;")
        video_layout = QVBoxLayout()
        video_layout.setContentsMargins(10, 10, 10, 10)
        video_layout.setSpacing(10)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("border: 2px solid #bdc3c7; border-radius: 10px;")
        self.video_label.setFixedSize(800, 600)

        self.count_label = QLabel("Adultos: 0 | Crianças: 0 | Animais: 0")
        self.count_label.setFont(QFont("Segoe UI", 12))
        self.count_label.setStyleSheet("color: #2c3e50;")
        self.count_label.setAlignment(Qt.AlignCenter)

        video_layout.addWidget(self.video_label)
        video_layout.addWidget(self.count_label)
        video_frame.setLayout(video_layout)

        # Adicionar widgets ao layout principal
        main_layout.addWidget(sidebar)
        main_layout.addWidget(video_frame)
        main_layout.setStretch(1, 1)

    def create_button(self, text, icon_path, callback, checkable=False):
        button = QPushButton(text)
        button.setFont(self.font)
        button.setIcon(QIcon(icon_path))
        button.setIconSize(QSize(24, 24))
        button.setCheckable(checkable)
        button.setStyleSheet("""
            QPushButton {
                background-color: #34495e;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                text-align: left;
            }
            QPushButton:hover {
                background-color: #3b5998;
            }
            QPushButton:checked {
                background-color: #16a085;
            }
        """)
        button.clicked.connect(callback)
        return button

    def connect_phone_camera(self):
        url, ok = QInputDialog.getText(
            self, 'URL da Câmera do Telefone', 'Digite o URL do DroidCam:'
        )
        if ok and url:
            self.video_source = url
            self.capture = None
            try:
                self.stream_generator = self.get_stream(url)
            except Exception as e:
                QMessageBox.critical(self, "Erro", f"Não foi possível conectar à câmera do telefone.\nErro: {e}")
        else:
            QMessageBox.warning(self, "Atenção", "Por favor, insira um URL válido.")

    def connect_webcam(self):
        self.video_source = 0
        self.capture = cv2.VideoCapture(self.video_source)
        self.stream_generator = None
        if not self.capture.isOpened():
            QMessageBox.critical(self, "Erro", "Não foi possível acessar a webcam.")

    def load_video_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, 'Selecionar Arquivo de Vídeo', '', 'Arquivos de Vídeo (*.mp4 *.avi *.mov)'
        )
        if file_path:
            self.video_source = file_path
            self.capture = cv2.VideoCapture(self.video_source)
            self.stream_generator = None
            if not self.capture.isOpened():
                QMessageBox.critical(self, "Erro", "Não foi possível abrir o arquivo de vídeo.")

    def toggle_bg_subtraction(self):
        self.effect_bg = self.btn_bg_subtraction.isChecked()
        self.update_effect_state()

    def toggle_object_detection(self):
        self.effect_detection = self.btn_object_detection.isChecked()
        self.update_effect_state()

    def update_effect_state(self):
        self.show_effect = self.effect_bg or self.effect_detection

    def stop_video(self):
        if self.capture:
            self.capture.release()
        self.video_label.clear()
        self.stream_generator = None
        self.count_label.setText("Adultos: 0 | Crianças: 0 | Animais: 0")
        self.btn_bg_subtraction.setChecked(False)
        self.btn_object_detection.setChecked(False)
        self.effect_bg = False
        self.effect_detection = False
        self.show_effect = False

    def get_stream(self, url):
        def stream():
            stream = requests.get(url, stream=True)
            bytes_data = b''
            for chunk in stream.iter_content(chunk_size=1024):
                bytes_data += chunk
                a = bytes_data.find(b'\xff\xd8')
                b = bytes_data.find(b'\xff\xd9')
                if a != -1 and b != -1:
                    jpg = bytes_data[a:b+2]
                    bytes_data = bytes_data[b+2:]
                    img_array = np.frombuffer(jpg, dtype=np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    if img is not None:
                        yield img
        return stream()

    def apply_bg_subtraction(self, frame):
        fgmask = self.fgbg.apply(frame)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
        return fgmask

    def apply_object_detection(self, frame):
        classes, scores, boxes = self.model.detect(frame, confThreshold=0.5, nmsThreshold=0.4)
        adultos, criancas, animais = 0, 0, 0

        for (classid, score, box) in zip(classes, scores, boxes):
            color = self.COLORS[int(classid) % len(self.COLORS)]
            label = f"{self.class_names[classid]}: {score:.2f}"
            cv2.rectangle(frame, box, color, 2)
            cv2.putText(
                frame, label, (box[0], box[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )

            # Heurística para distinguir adultos de crianças baseado na altura da bounding box
            if self.class_names[classid] == "person":
                _, _, w, h = box
                if h > 150:  # Ajuste este valor conforme necessário
                    adultos += 1
                else:
                    criancas += 1
            elif self.class_names[classid] in ["cat", "dog"]:
                animais += 1

        # Atualizar o rótulo de contagem
        self.count_label.setText(f"Adultos: {adultos} | Crianças: {criancas} | Animais: {animais}")
        return frame

    def update_frame(self):
        if self.stream_generator:
            try:
                frame = next(self.stream_generator)
            except StopIteration:
                self.stop_video()
                return
            except Exception as e:
                QMessageBox.critical(self, "Erro", f"Erro ao receber frame da câmera do telefone.\nErro: {e}")
                self.stop_video()
                return
        elif self.capture and self.capture.isOpened():
            ret, frame = self.capture.read()
            if not ret:
                self.stop_video()
                return
        else:
            return

        if frame is not None:
            if self.show_effect:
                if self.effect_bg:
                    frame = self.apply_bg_subtraction(frame)
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                if self.effect_detection:
                    frame = self.apply_object_detection(frame)

            frame = cv2.resize(frame, (800, 600))
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            qimg = QImage(
                image.data, image.shape[1], image.shape[0],
                image.strides[0], QImage.Format_RGB888
            )
            self.video_label.setPixmap(QPixmap.fromImage(qimg))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    janela = AplicativoDeVideo()
    janela.show()
    sys.exit(app.exec_())
