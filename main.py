import cv2
import numpy as np
import pygame
import threading
import matplotlib.pyplot as plt

# === CONFIGURACIÓN ===
ALTURA_REAL_CM = 14  # altura real del vaso

pygame.mixer.init()

# === FUNCIONES ===
def reproducir_sonido(ruta):
    def _play():
        sonido = pygame.mixer.Sound(ruta)
        sonido.play()
    threading.Thread(target=_play, daemon=True).start()

# === VARIABLES DE ESTADO ===
puntos = []
base_y = None
top_y = None
pixeles_por_cm = None
calibrado = False
riesgo_actual = None

# === CONFIGURAR GRÁFICO ===
niveles = [0.0, 0.3, 0.5, 1.5, 2.5, 3.0]
riesgos = [0, 1, 2, 3, 4, 5]
etiquetas = [
    "Riesgo nulo",
    "Riesgo medio",
    "Riesgo alto",
    "Riesgo muy alto",
    "Riesgo severo",
    "Riesgo extremo"
]

plt.ion()
fig, ax = plt.subplots()
ax.plot(niveles, riesgos, label="Riesgo", color="blue")
point, = ax.plot(0, 0, 'ro', label="Nivel actual")
ax.set_xlabel("Alcoholemia (g/L)")
ax.set_ylabel("Nivel de riesgo")
ax.set_ylim(-0.5, 5.5)
ax.set_xlim(0, 3.1)
ax.set_yticks(riesgos)
ax.set_yticklabels(etiquetas)
mensaje_grafico = ax.text(0.05, 5.2, "", fontsize=10, color='red')
plt.title("Nivel de alcoholemia y riesgo")
plt.legend()
plt.draw()

# === FUNCIONES DE EVENTO ===
def click_event(event, x, y, flags, param):
    global puntos, base_y, top_y, pixeles_por_cm, calibrado
    if event == cv2.EVENT_LBUTTONDOWN:
        puntos.append((x, y))
        if len(puntos) == 1:
            base_y = y
            print(f"Base marcada en Y = {base_y}")
        elif len(puntos) == 2:
            top_y = y
            print(f"Tope marcado en Y = {top_y}")
            altura_pixels = abs(base_y - top_y)
            pixeles_por_cm = altura_pixels / ALTURA_REAL_CM
            calibrado = True
            print(f"Calibración lista. {pixeles_por_cm:.2f} px/cm")

# === CAPTURA DE VIDEO ===
cap = cv2.VideoCapture(0)
cv2.namedWindow("Detección")
cv2.setMouseCallback("Detección", click_event)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error accediendo a la cámara.")
        break

    frame = cv2.flip(frame, 1)  # espejo horizontal

    if calibrado:
        # Convertir a HSV para mejor segmentación
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Rango de rojo en HSV
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        # Encontrar contornos
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            c = max(contours, key=cv2.contourArea)
            (x, y, w, h) = cv2.boundingRect(c)
            center_y = y + h // 2
            altura_cm = (base_y - center_y) / pixeles_por_cm
            alcoholemia = max(0.0, min((altura_cm / ALTURA_REAL_CM) * 3.0, 3.0))

            # Determinar riesgo y mensaje
            riesgo_nuevo = None
            mensaje = ""
            sonido = None

            if alcoholemia < 0.01:
                riesgo_nuevo = 0
                mensaje = "0.0 g/L - Riesgo nulo. Conduce responsablemente."
            elif alcoholemia < 0.3:
                riesgo_nuevo = 1
                mensaje = f"{alcoholemia:.2f} g/L - Riesgo medio."
                sonido = "tomando.wav"
            elif alcoholemia < 0.5:
                riesgo_nuevo = 2
                mensaje = f"{alcoholemia:.2f} g/L - Riesgo alto."
                sonido = "tomando.wav"
            elif alcoholemia < 1.5:
                riesgo_nuevo = 3
                mensaje = f"{alcoholemia:.2f} g/L - Riesgo muy alto."
                sonido = "fallout.wav"
            elif alcoholemia < 2.5:
                riesgo_nuevo = 4
                mensaje = f"{alcoholemia:.2f} g/L - Riesgo severo."
                sonido = "fallout.wav"
            else:
                riesgo_nuevo = 5
                mensaje = f"{alcoholemia:.2f} g/L - Riesgo extremo. Peligro de muerte."
                sonido = "sirena.wav"

            # Reproducir sonido si cambió el riesgo
            if riesgo_nuevo != riesgo_actual:
                riesgo_actual = riesgo_nuevo
                if sonido:
                    reproducir_sonido(sonido)

            # Dibujar en imagen
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.line(frame, (0, base_y), (frame.shape[1], base_y), (255, 255, 255), 2)
            cv2.line(frame, (0, top_y), (frame.shape[1], top_y), (200, 200, 200), 1)
            cv2.line(frame, (0, center_y), (frame.shape[1], center_y), (0, 0, 255), 1)
            cv2.putText(frame, mensaje, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            # Actualizar gráfico
            riesgo_interp = np.interp(alcoholemia, niveles, riesgos)
            point.set_data([alcoholemia], [riesgo_interp])
            mensaje_grafico.set_text(mensaje)
            plt.draw()
            plt.pause(0.001)

    else:
        cv2.putText(frame, "Haz clic en la base y luego en el tope del vaso", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("Detección", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.close()
