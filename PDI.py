import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QFileDialog, QInputDialog, QHBoxLayout, QLabel, QGridLayout
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import QTimer, Qt

class AplicativoDeVideo(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Detecção de Vídeo')  # Título da janela
        self.setGeometry(100, 100, 1200, 800)  # Dimensões da janela
        self.video_source = None
        self.capture = None
        self.stream_generator = None
        self.show_effect = False
        self.effect_bg = False
        self.effect_detection = False

        # Carregar o modelo YOLOv4-tiny
        self.net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
        self.model = cv2.dnn.DetectionModel(self.net)
        self.model.setInputParams(size=(416, 416), scale=1/255)

        # Carregar as classes COCO
        self.class_names = []
        with open("coco.names", "r") as f:
            self.class_names = [cname.strip() for cname in f.readlines()]

        # Cores para as classes
        self.CORES = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

        # Inicializar o Subtrator de Fundo
        self.fgbg = cv2.createBackgroundSubtractorMOG2()

        # Criar o widget central e layout principal
        self.widget_central = QWidget()
        self.setCentralWidget(self.widget_central)
        self.layout_principal = QGridLayout()
        self.widget_central.setLayout(self.layout_principal)

        # Criar a barra lateral para navegação
        self.widget_lateral = QWidget()
        self.layout_lateral = QVBoxLayout()
        self.layout_lateral.setContentsMargins(10, 10, 10, 10)
        self.layout_lateral.setSpacing(10)
        self.widget_lateral.setLayout(self.layout_lateral)
        self.layout_principal.addWidget(self.widget_lateral, 0, 0)

        # Criar o rótulo para exibição do vídeo
        self.rotulo_video = QLabel()
        self.rotulo_video.setStyleSheet("border: 2px solid #34495e;")  # Estilo da borda atualizado
        self.layout_principal.addWidget(self.rotulo_video, 0, 1)

        # Criar o rótulo para mostrar as contagens
        self.rotulo_contagem = QLabel("Adultos: 0 | Crianças: 0 | Animais: 0")
        fonte = QFont()
        fonte.setPointSize(14)
        self.rotulo_contagem.setFont(fonte)
        self.layout_principal.addWidget(self.rotulo_contagem, 1, 1)

        # Inicializar o temporizador para atualização do vídeo
        self.temporizador = QTimer()
        self.temporizador.timeout.connect(self.atualizar_frame)
        self.temporizador.start(30)

        # Criar os botões da barra lateral com ícones
        self.criar_botoes_barra_lateral()

    def criar_botoes_barra_lateral(self):
        estilo_botao = "background-color: #3498db; color: white; border: none; padding: 10px; border-radius: 5px;"
        estilo_efeito = "background-color: #2ecc71; color: white; border: none; padding: 10px; border-radius: 5px;"
        
        # Botão da Câmera do Telefone
        self.botao_telefone = QPushButton('Câmera do Telefone')
        self.botao_telefone.setStyleSheet(estilo_botao)
        self.botao_telefone.clicked.connect(self.conectar_telefone)
        self.layout_lateral.addWidget(self.botao_telefone)

        # Botão da Webcam
        self.botao_webcam = QPushButton('Webcam')
        self.botao_webcam.setStyleSheet(estilo_botao)
        self.botao_webcam.clicked.connect(self.conectar_webcam)
        self.layout_lateral.addWidget(self.botao_webcam)

        # Botão do Arquivo de Vídeo
        self.botao_video = QPushButton('Arquivo de Vídeo')
        self.botao_video.setStyleSheet(estilo_botao)
        self.botao_video.clicked.connect(self.carregar_video)
        self.layout_lateral.addWidget(self.botao_video)

        # Botões de Efeito
        self.botao_sub_bg = QPushButton('Subtração de Fundo')
        self.botao_sub_bg.setStyleSheet(estilo_efeito)
        self.botao_sub_bg.clicked.connect(self.alternar_subtracao_fundo)
        self.layout_lateral.addWidget(self.botao_sub_bg)
        
        self.botao_det_obj = QPushButton('Detecção de Objetos')
        self.botao_det_obj.setStyleSheet("background-color: #e67e22; color: white; border: none; padding: 10px; border-radius: 5px;")
        self.botao_det_obj.clicked.connect(self.alternar_deteccao_objetos)
        self.layout_lateral.addWidget(self.botao_det_obj)
        
        self.botao_parar = QPushButton('Parar')
        self.botao_parar.setStyleSheet("background-color: #e74c3c; color: white; border: none; padding: 10px; border-radius: 5px;")
        self.botao_parar.clicked.connect(self.parar_video)
        self.layout_lateral.addWidget(self.botao_parar)

        # Configurar propriedades da barra lateral
        self.widget_lateral.setFixedWidth(200)
        self.widget_lateral.setStyleSheet("background-color: #2c3e50;")

    def conectar_telefone(self):
        url, ok = QInputDialog.getText(self, 'URL da Câmera do Telefone', 'Digite o URL do DroidCam:')
        if ok and url:
            self.video_source = url
            self.capture = None
            self.stream_generator = self.obter_stream(url)
        else:
            print("Por favor, insira um URL válido.")

    def conectar_webcam(self):
        self.video_source = 0  # Webcam padrão
        self.capture = cv2.VideoCapture(self.video_source)
        self.stream_generator = None

    def carregar_video(self):
        caminho_arquivo, _ = QFileDialog.getOpenFileName(self, 'Selecionar Arquivo de Vídeo', '', 'Arquivos de Vídeo (*.mp4 *.avi)')
        if caminho_arquivo:
            self.video_source = caminho_arquivo
            self.capture = cv2.VideoCapture(self.video_source)
            self.stream_generator = None

    def alternar_subtracao_fundo(self):
        self.effect_bg = not self.effect_bg
        self.show_effect = self.effect_bg or self.effect_detection

    def alternar_deteccao_objetos(self):
        self.effect_detection = not self.effect_detection
        self.show_effect = self.effect_detection or self.effect_bg

    def parar_video(self):
        if self.capture:
            self.capture.release()
        self.rotulo_video.clear()
        self.stream_generator = None

    def obter_stream(self, url):
        import requests
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

    def aplicar_efeito_subtracao_fundo(self, frame):
        fgmask = self.fgbg.apply(frame)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        return fgmask

    def aplicar_efeito_deteccao_objetos(self, frame):
        classes, scores, boxes = self.model.detect(frame, 0.1, 0.2)
        adultos, criancas, animais = 0, 0, 0

        for (classid, score, box) in zip(classes, scores, boxes):
            cor = self.CORES[int(classid) % len(self.CORES)]
            rotulo = f"{self.class_names[classid]} : {score:.2f}"
            cv2.rectangle(frame, box, cor, 2)
            cv2.putText(frame, rotulo, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 2)

            # Incrementar contagem com base na classe
            if self.class_names[classid] in ["person"]:  # Assumindo que 'person' representa adultos
                adultos += 1
            elif self.class_names[classid] in ["cat", "dog"]:  # Assumindo que 'cat' e 'dog' representam animais
                animais += 1
            # Estender conforme necessário para crianças se as classes estiverem disponíveis

        # Atualizar o rótulo de contagem
        self.rotulo_contagem.setText(f"Adultos: {adultos} | Crianças: 0 | Animais: {animais}")
        return frame

    def atualizar_frame(self):
        if self.stream_generator:
            frame = next(self.stream_generator)
        elif self.capture and self.capture.isOpened():
            ret, frame = self.capture.read()
            if not ret:
                return
        else:
            return

        if frame is not None:
            # Redimensionar o frame para se ajustar ao espaço disponível
            h, w = frame.shape[:2]
            novo_w, novo_h = 800, int(h * 800 / w)
            frame = cv2.resize(frame, (novo_w, novo_h))
            
            # Converter o frame para o formato RGB para exibição
            imagem_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = imagem_rgb.shape
            bytes_por_linha = ch * w
            qimg = QImage(imagem_rgb.data, w, h, bytes_por_linha, QImage.Format_RGB888)
            self.rotulo_video.setPixmap(QPixmap.fromImage(qimg))

            if self.show_effect:
                if self.effect_bg:
                    fgmask = self.aplicar_efeito_subtracao_fundo(frame)
                    imagem_bg = QImage(fgmask.data, fgmask.shape[1], fgmask.shape[0], fgmask.strides[0], QImage.Format_Grayscale8)
                    self.rotulo_video.setPixmap(QPixmap.fromImage(imagem_bg))

                if self.effect_detection:
                    frame_deteccao = self.aplicar_efeito_deteccao_objetos(frame)
                    imagem_rgb_deteccao = cv2.cvtColor(frame_deteccao, cv2.COLOR_BGR2RGB)
                    qimg_deteccao = QImage(imagem_rgb_deteccao.data, imagem_rgb_deteccao.shape[1], imagem_rgb_deteccao.shape[0], imagem_rgb_deteccao.strides[0], QImage.Format_RGB888)
                    self.rotulo_video.setPixmap(QPixmap.fromImage(qimg_deteccao))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    janela = AplicativoDeVideo()
    janela.show()
    sys.exit(app.exec_())
