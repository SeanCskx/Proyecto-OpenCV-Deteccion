import cv2

# 1. Cargar los clasificadores en cascada de Haar
# Asegúrate de tener OpenCV instalado (pip install opencv-python)
# OpenCV ya incluye estos archivos, por lo que no necesitas descargarlos por separado.
try:
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
    smile_cascade_path = cv2.data.haarcascades + 'haarcascade_smile.xml'

    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
    smile_cascade = cv2.CascadeClassifier(smile_cascade_path)

    if face_cascade.empty() or eye_cascade.empty() or smile_cascade.empty():
        raise IOError("No se pudieron cargar uno o más clasificadores de Haar.")
    
    print("Clasificadores cargados correctamente.")

except Exception as e:
    print(e)
    exit()

# 2. Activar la cámara
# El índice 0 corresponde a la cámara web por defecto. Si tienes varias, puedes probar con 1, 2, etc.
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Error: No se puede abrir la cámara.")
    exit()

print("Cámara activada. Presiona 'q' en la ventana de video para salir.")

# 3. Bucle principal para procesar cada fotograma del video
while True:
    # Leer un fotograma de la cámara
    ret, frame = cam.read()
    if not ret:
        print("No se pudo recibir el fotograma. Saliendo...")
        break

    # Convertir el fotograma a escala de grises para la detección
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar rostros en el fotograma
    rostros = face_cascade.detectMultiScale(gris, 1.3, 5)

    # Para cada rostro detectado, buscar ojos y sonrisas
    for (x, y, w, h) in rostros:
        # Dibujar rectángulo verde para el rostro
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Definir la Región de Interés (ROI) para buscar dentro del rostro
        roi_gris = gris[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Detectar ojos dentro del rostro
        ojos = eye_cascade.detectMultiScale(roi_gris, 1.1, 22)
        for (ex, ey, ew, eh) in ojos:
            # Dibujar rectángulo azul para los ojos
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)

        # Detectar sonrisas dentro del rostro
        sonrisas = smile_cascade.detectMultiScale(roi_gris, 1.8, 20)
        for (sx, sy, sw, sh) in sonrisas:
             # Dibujar rectángulo rojo para las sonrisas
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)

    # Mostrar el fotograma con las detecciones en una ventana
    cv2.imshow("Deteccion en Vivo - Presiona 'q' para salir", frame)

    # Esperar por la tecla 'q' para salir del bucle
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 4. Liberar recursos
print("Cerrando la aplicación...")
cam.release()
cv2.destroyAllWindows()
