import cv2
import numpy as np
import math

ORIENTACOES = ["anticlockwise", "clockwise", "anticlockwise", "clockwise"]

def load_cv2_from_bytes(img_bytes):
    arr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def recortar_dials(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 2)
    h, w = img.shape[:2]
    min_r = int(min(w, h) * 0.08)
    max_r = int(min(w, h) * 0.30)

    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=min_r * 2,
        param1=100, param2=40,
        minRadius=min_r, maxRadius=max_r
    )

    dials = []
    if circles is not None:
        circles = np.round(circles[0]).astype(int)
        circles = sorted(circles, key=lambda c: c[0])
        if len(circles) >= 4:
            for cx, cy, r in circles[:4]:
                m = int(r * 1.1)
                x1 = max(cx - m, 0); y1 = max(cy - m, 0)
                x2 = min(cx + m, w); y2 = min(cy + m, h)
                dials.append(img[y1:y2, x1:x2])
            return dials

    # fallback: divide em 4 faixas horizontais
    dw = w // 4
    for i in range(4):
        x1 = i * dw
        x2 = (i + 1) * dw if i < 3 else w
        dials.append(img[0:h, x1:x2])
    return dials

def detectar_angulo(dial_img):
    gray = cv2.cvtColor(dial_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    edges = cv2.Canny(gray, 40, 120)

    h, w = gray.shape[:2]
    cx, cy = w // 2, h // 2

    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=20,
        minLineLength=int(min(w, h) * 0.25),
        maxLineGap=15
    )

    if lines is None:
        return None

    best_angle = None
    best_score = -1

    for line in lines:
        x1, y1, x2, y2 = line[0]
        d1 = math.hypot(x1 - cx, y1 - cy)
        d2 = math.hypot(x2 - cx, y2 - cy)

        if d2 > d1:
            px, py = x2, y2
            near_d = d1
        else:
            px, py = x1, y1
            near_d = d2

        far_d = math.hypot(px - cx, py - cy)
        score = far_d - near_d * 2

        if score > best_score:
            best_score = score
            vx = px - cx
            vy = cy - py
            ang = math.degrees(math.atan2(vx, vy))
            best_angle = ang

    if best_angle is None:
        return None

    return best_angle % 360

def angulo_para_digito(angle, orientation):
    if orientation == "anticlockwise":
        angle = (360 - angle) % 360
    return int((angle + 18) % 360 // 36) % 10

def ler_medidor(img_bytes):
    img = load_cv2_from_bytes(img_bytes)
    if img is None:
        return {"erro": "Imagem inválida"}

    dials = recortar_dials(img)
    digitos = []
    angulos = []

    for dial_img, orient in zip(dials, ORIENTACOES):
        angle = detectar_angulo(dial_img)
        angulos.append(round(angle, 1) if angle is not None else None)
        if angle is None:
            digitos.append(None)
        else:
            digitos.append(angulo_para_digito(angle, orient))

    if any(d is None for d in digitos):
        return {
            "erro": "Não detectei todos os ponteiros",
            "digitos": digitos,
            "angulos": angulos
        }

    leitura = int("".join(str(d) for d in digitos))
    return {"digitos": digitos, "angulos": angulos, "leitura_kwh": leitura}