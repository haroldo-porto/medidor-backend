import cv2
import numpy as np
import math

ORIENTACOES = ["anticlockwise", "clockwise", "anticlockwise", "clockwise"]


def load_cv2_from_bytes(img_bytes):
    arr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


# ─────────────────────────────────────────────
# PRÉ-PROCESSAMENTO: converte para P&B e realça
# ─────────────────────────────────────────────
def preprocessar(img):
    # 1. Converte para escala de cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Remove reflexo com desfoque bilateral
    #    (preserva bordas enquanto suaviza brilhos)
    gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    # 3. CLAHE: aumenta contraste local
    #    (faz o ponteiro escuro se destacar do fundo claro)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
    gray = clahe.apply(gray)

    # 4. Threshold adaptativo
    #    (binariza: tudo fica preto ou branco baseado na região local)
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=21,
        C=8
    )

    # 5. Remove ruído pequeno (pontos isolados)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return gray, binary


# ─────────────────────────────────────────────
# RECORTAR OS 4 DIALS
# ─────────────────────────────────────────────
def recortar_dials(img, gray):
    h, w = img.shape[:2]
    blur = cv2.GaussianBlur(gray, (7, 7), 2)

    min_r = int(min(w, h) * 0.07)
    max_r = int(min(w, h) * 0.28)

    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.0,
        minDist=int(min(w, h) * 0.14),
        param1=60,
        param2=30,
        minRadius=min_r,
        maxRadius=max_r
    )

    dials_color = []
    dials_binary = []
    centros = []

    _, binary = preprocessar(img)

    if circles is not None:
        circles = np.round(circles[0]).astype(int)
        circles = sorted(circles, key=lambda c: c[0])

        # Filtra círculos muito próximos
        filtrados = [circles[0]]
        for c in circles[1:]:
            if abs(c[0] - filtrados[-1][0]) > min_r * 1.5:
                filtrados.append(c)
            if len(filtrados) == 4:
                break

        if len(filtrados) == 4:
            for cx, cy, r in filtrados:
                pad = int(r * 1.2)
                x1 = max(cx - pad, 0); y1 = max(cy - pad, 0)
                x2 = min(cx + pad, w); y2 = min(cy + pad, h)
                dials_color.append(img[y1:y2, x1:x2])
                dials_binary.append(binary[y1:y2, x1:x2])
                centros.append((cx, cy, r))
            return dials_color, dials_binary, centros

    # Fallback: divide em 4 faixas horizontais
    dw = w // 4
    for i in range(4):
        x1 = i * dw
        x2 = (i + 1) * dw if i < 3 else w
        dials_color.append(img[0:h, x1:x2])
        dials_binary.append(binary[0:h, x1:x2])
        cx = (x1 + x2) // 2
        centros.append((cx, h // 2, min(dw, h) // 2))

    return dials_color, dials_binary, centros


# ─────────────────────────────────────────────
# DETECTAR ÂNGULO DO PONTEIRO (usa imagem binária)
# ─────────────────────────────────────────────
def detectar_angulo(dial_color, dial_binary):
    h, w = dial_binary.shape[:2]
    cx, cy = w // 2, h // 2
    r = min(w, h) // 2

    # Cria máscara circular (ignora bordas do mostrador)
    mask_circ = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask_circ, (cx, cy), int(r * 0.85), 255, -1)

    # Remove centro (pivot do ponteiro causa ruído)
    cv2.circle(mask_circ, (cx, cy), int(r * 0.15), 0, -1)

    # Aplica máscara na imagem binária
    dial_masked = cv2.bitwise_and(dial_binary, mask_circ)

    # ── Tentativa 1: detectar ponteiro vermelho ──
    angle = detectar_vermelho(dial_color, cx, cy, r)
    if angle is not None:
        return angle, "vermelho"

    # ── Tentativa 2: usar imagem binária com Hough Lines ──
    angle = detectar_por_linhas(dial_masked, cx, cy, r)
    if angle is not None:
        return angle, "binario"

    # ── Tentativa 3: usar momento dos pixels (centro de massa) ──
    angle = detectar_por_momento(dial_masked, cx, cy, r)
    if angle is not None:
        return angle, "momento"

    return None, None


def detectar_vermelho(dial_color, cx, cy, r):
    hsv = cv2.cvtColor(dial_color, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, np.array([0, 120, 80]), np.array([8, 255, 255]))
    m2 = cv2.inRange(hsv, np.array([168, 120, 80]), np.array([180, 255, 255]))
    mask = cv2.bitwise_or(m1, m2)

    # Remove área central
    cv2.circle(mask, (cx, cy), int(r * 0.15), 0, -1)

    pts = cv2.findNonZero(mask)
    if pts is None or len(pts) < 8:
        return None

    pts = pts.reshape(-1, 2).astype(np.float32)
    dists = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)

    # Usa apenas os 30% mais distantes do centro (ponta do ponteiro)
    threshold = np.percentile(dists, 70)
    far = pts[dists >= threshold]

    if len(far) < 3:
        return None

    mx = np.mean(far[:, 0]) - cx
    my = cy - np.mean(far[:, 1])
    return math.degrees(math.atan2(mx, my)) % 360


def detectar_por_linhas(dial_binary, cx, cy, r):
    min_len = int(r * 0.35)

    lines = cv2.HoughLinesP(
        dial_binary, 1, np.pi / 360,
        threshold=12,
        minLineLength=min_len,
        maxLineGap=8
    )

    if lines is None:
        return None

    candidatos = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        d1 = math.hypot(x1 - cx, y1 - cy)
        d2 = math.hypot(x2 - cx, y2 - cy)

        if d2 > d1:
            px, py = x2, y2
            near_d = d1
            far_d = d2
        else:
            px, py = x1, y1
            near_d = d2
            far_d = d1

        # O ponteiro passa pelo centro: near_d deve ser pequeno
        if near_d > r * 0.45:
            continue

        comprimento = math.hypot(x2 - x1, y2 - y1)
        score = far_d * comprimento / max(near_d + 1, 1)
        vx = px - cx
        vy = cy - py
        ang = math.degrees(math.atan2(vx, vy)) % 360
        candidatos.append((score, ang))

    if not candidatos:
        return None

    # Retorna o ângulo com maior score
    candidatos.sort(reverse=True)
    return candidatos[0][1]


def detectar_por_momento(dial_binary, cx, cy, r):
    """
    Divide o dial em 36 fatias de 10° e conta pixels em cada uma.
    A fatia com mais pixels indica a direção do ponteiro.
    """
    h, w = dial_binary.shape[:2]
    contagens = []

    for ang_deg in range(0, 360, 10):
        ang_rad = math.radians(ang_deg)
        mascara = np.zeros((h, w), dtype=np.uint8)

        # Cria um triângulo fino na direção do ângulo
        p1 = (cx, cy)
        ang1 = ang_rad - math.radians(8)
        ang2 = ang_rad + math.radians(8)
        p2 = (int(cx + r * 0.9 * math.sin(ang1)),
               int(cy - r * 0.9 * math.cos(ang1)))
        p3 = (int(cx + r * 0.9 * math.sin(ang2)),
               int(cy - r * 0.9 * math.cos(ang2)))
        cv2.fillPoly(mascara, [np.array([p1, p2, p3])], 255)

        overlap = cv2.bitwise_and(dial_binary, mascara)
        contagens.append(cv2.countNonZero(overlap))

    melhor = int(np.argmax(contagens)) * 10
    return float(melhor)


# ─────────────────────────────────────────────
# ÂNGULO → DÍGITO
# ─────────────────────────────────────────────
def angulo_para_digito(angle, orientation):
    if orientation == "anticlockwise":
        angle = (360 - angle) % 360
    return int((angle + 18) % 360 // 36) % 10


# ─────────────────────────────────────────────
# CALIBRAÇÃO ENTRE DIALS ADJACENTES
# ─────────────────────────────────────────────
def calibrar_digitos(digitos, angulos):
    resultado = list(digitos)
    for i in range(len(digitos) - 1):
        ang = angulos[i]
        if ang is None:
            continue
        orient = ORIENTACOES[i]
        ang_norm = (360 - ang) % 360 if orient == "anticlockwise" else ang
        pos = ang_norm % 36
        proximo = digitos[i + 1] if i + 1 < len(digitos) else None
        if proximo is not None:
            if pos >= 31:
                resultado[i] = digitos[i] if proximo < 5 else (digitos[i] + 1) % 10
            elif pos <= 5:
                resultado[i] = digitos[i] if proximo >= 5 else (digitos[i] - 1) % 10
    return resultado


# ─────────────────────────────────────────────
# FUNÇÃO PRINCIPAL
# ─────────────────────────────────────────────
def ler_medidor(img_bytes):
    img = load_cv2_from_bytes(img_bytes)
    if img is None:
        return {"erro": "Imagem inválida ou corrompida."}

    # Redimensiona se muito grande
    h, w = img.shape[:2]
    if w > 1400:
        scale = 1400 / w
        img = cv2.resize(img, (1400, int(h * scale)))

    # Pré-processamento P&B
    gray, binary = preprocessar(img)

    # Recorta os 4 dials
    dials_color, dials_binary, centros = recortar_dials(img, gray)

    if len(dials_color) < 4:
        return {"erro": "Não encontrei 4 dials. Tente uma foto mais próxima e centralizada."}

    digitos = []
    angulos = []
    metodos = []

    for dial_color, dial_bin, orient in zip(dials_color, dials_binary, ORIENTACOES):
        angle, metodo = detectar_angulo(dial_color, dial_bin)
        angulos.append(round(angle, 1) if angle is not None else None)
        metodos.append(metodo)
        digitos.append(angulo_para_digito(angle, orient) if angle is not None else None)

    if any(d is None for d in digitos):
        return {
            "erro": "Não detectei todos os ponteiros. Tente foto mais iluminada e sem reflexo.",
            "digitos": digitos,
            "angulos": angulos,
            "metodos": metodos
        }

    digitos_finais = calibrar_digitos(digitos, angulos)
    leitura = int("".join(str(d) for d in digitos_finais))

    return {
        "digitos": digitos_finais,
        "digitos_brutos": digitos,
        "angulos": angulos,
        "metodos": metodos,
        "leitura_kwh": leitura
    }
