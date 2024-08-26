import requests
import numpy as np
import cv2
import io

# Substitua pelo endereço IP e porta fornecidos pelo DroidCam
url = 'http://192.168.2.102:4747/video'

# Função para converter imagem JPEG para formato OpenCV
def jpeg_to_opencv(jpeg_data):
    img_array = np.frombuffer(jpeg_data, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img

# Inicializa o subtrator de fundo
fgbg = cv2.createBackgroundSubtractorMOG2()

# Configura uma requisição para o stream com timeout
try:
    stream = requests.get(url, stream=True, timeout=10)
    print("Conectado ao stream.")
except requests.exceptions.RequestException as e:
    print(f"Erro ao conectar: {e}")
    exit()

bytes_data = b''
while True:
    for chunk in stream.iter_content(chunk_size=1024):
        bytes_data += chunk
        a = bytes_data.find(b'\xff\xd8')  # Início do JPEG
        b = bytes_data.find(b'\xff\xd9')  # Fim do JPEG
        if a != -1 and b != -1:
            jpg = bytes_data[a:b+2]
            bytes_data = bytes_data[b+2:]
            frame = jpeg_to_opencv(jpg)
            if frame is not None:
                # Aplicar subtração de fundo
                fgmask = fgbg.apply(frame)

                # Remover ruído usando operações morfológicas
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
                fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

                # Encontrar contornos dos objetos em movimento
                contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Desenhar retângulos em torno dos objetos detectados
                for contour in contours:
                    if cv2.contourArea(contour) > 500:  # Filtra objetos muito pequenos
                        x, y, w, h = cv2.boundingRect(contour)
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # Exibir o vídeo com os objetos detectados
                cv2.imshow('Frame', frame)
                cv2.imshow('FG Mask', fgmask)

                # Encerrar com a tecla 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

# Libere os recursos
stream.close()
cv2.destroyAllWindows()
