import cv2
import numpy as np
import math

# Medidores tipo relógio: dials alternam sentido
# 1º = anti-horário, 2º = horário, 3º = anti-horário, 4º = horário
ORIENTACOES = ["anticlockwise", "clockwise", "anticlockwise", "clockwise"]


def load_cv2_from_bytes(img_bytes):
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


# ─────────────────────────────────────────────
# 1. DETECTAR E RECORTAR OS 4 DIALS
# ─────────────────────────────────────────────
def recortar_dials(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 2)
    h, w = img.shape[:2]

    # Tenta detectar círculos
    min_r = int(min(w, h) * 0.07)
    max_r = int(min(w, h) * 0.28)

    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.0,
        minDist=int(min(w, h) * 0.15),
        param1=80,
        param2=35,
        minRadius=min_r,
        maxRadius=max_r,
    )

    dials = []
    centros = []

    if circles is not None:
        circles = np.round(circles[0]).astype(int)
        # Ordena da esquerda para direita
        circles = sorted(circles, key=lambda c: c[0])

        # Pega os 4 círculos mais prováveis (filtra os muito parecidos)
        filtrados = [circles[0]]
        for c in circles[1:]:
            if abs(c[0] - filtrados[-1][0]) > min_r:
                filtrados.append(c)
            if len(filtrados) == 4:
                break

        if len(filtrados) == 4:
            for cx, cy, r in filtrados:
                pad = int(r * 1.15)
                x1 = max(cx - pad, 0)
                y1 = max(cy - pad, 0)
                x2 = min(cx + pad, w)
                y2 = min(cy + pad, h)
                dials.append(img[y1:y2, x1:x2])
                centros.append((cx, cy, r))
            return dials, centros

    # Fallback: divide horizontalmente em 4 partes iguais
    dw = w // 4
    for i in range(4):
        x1 = i * dw
        x2 = (i + 1) * dw if i < 3 else w
        cx = (x1 + x2) // 2
        cy = h // 2
        r = min(dw, h) // 2
        dials.append(img[0:h, x1:x2])
        centros.append((cx, cy, r))

    return dials, centros


# ─────────────────────────────────────────────
# 2. DETECTAR A COR VERMELHA DO PONTEIRO
# ─────────────────────────────────────────────
def mascara_vermelha(dial_img):
    hsv = cv2.cvtColor(dial_img, cv2.COLOR_BGR2HSV)

    # Vermelho tem duas faixas no HSV
    m1 = cv2.inRange(hsv, np.array([0, 100, 80]), np.array([10, 255, 255]))
    m2 = cv2.inRange(hsv, np.array([165, 100, 80]), np.array([180, 255, 255]))
    mask = cv2.bitwise_or(m1, m2)

    # Remove ruído
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)

    return mask


# ─────────────────────────────────────────────
# 3. CALCULAR ÂNGULO DO PONTEIRO
# ─────────────────────────────────────────────
def angulo_do_ponteiro(dial_img):
    h, w = dial_img.shape[:2]
    cx, cy = w // 2, h // 2

    # ── Tentativa 1: detectar pela cor vermelha ──
    mask = mascara_vermelha(dial_img)
    pixels_vermelhos = cv2.countNonZero(mask)

    if pixels_vermelhos > 20:
        angle = angulo_pela_mascara(mask, cx, cy)
        if angle is not None:
            return angle, "red"

    # ── Tentativa 2: usar edges gerais (ponteiro escuro/preto) ──
    gray = cv2.cvtColor(dial_img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    gray = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 30, 100)

    # Mascara circular para ignorar bordas do mostrador
    circle_mask = np.zeros((h, w), dtype=np.uint8)
    radius = int(min(w, h) * 0.42)
    cv2.circle(circle_mask, (cx, cy), radius, 255, -1)

    # Remove área central (pivot do ponteiro)
    cv2.circle(circle_mask, (cx, cy), int(radius * 0.12), 0, -1)

    edges = cv2.bitwise_and(edges, circle_mask)

    angle = angulo_pelas_linhas(edges, cx, cy, h, w)
    return angle, "edge"


def angulo_pela_mascara(mask, cx, cy):
    """Calcula ângulo usando os pixels vermelhos detectados."""
    h, w = mask.shape[:2]

    # Remove área central (pivot)
    centro_mask = np.zeros_like(mask)
    cv2.circle(centro_mask, (cx, cy), int(min(w, h) * 0.10), 255, -1)
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(centro_mask))

    coords = cv2.findNonZero(mask)
    if coords is None or len(coords) < 5:
        return None

    coords = coords.reshape(-1, 2).astype(np.float32)

    # Pega os pixels mais distantes do centro (ponta do ponteiro)
    dists = np.sqrt((coords[:, 0] - cx) ** 2 + (coords[:, 1] - cy) ** 2)
    idx = np.argsort(dists)[-max(5, len(coords) // 3):]
    far_pts = coords[idx]

    # Vetor médio da ponta
    mean_x = np.mean(far_pts[:, 0]) - cx
    mean_y = cy - np.mean(far_pts[:, 1])  # inverte Y

    angle = math.degrees(math.atan2(mean_x, mean_y)) % 360
    return angle


def angulo_pelas_linhas(edges, cx, cy, h, w):
    """Detecta o ponteiro via Hough Lines."""
    min_len = int(min(w, h) * 0.22)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=15,
        minLineLength=min_len,
        maxLineGap=10,
    )

    if lines is None:
        return None

    best_angle = None
    best_score = -1

    for line in lines:
        x1, y1, x2, y2 = line[0]

        d1 = math.hypot(x1 - cx, y1 - cy)
        d2 = math.hypot(x2 - cx, y2 - cy)

        # O ponto mais distante é a ponta do ponteiro
        if d2 > d1:
            px, py = x2, y2
            near_d = d1
            far_d = d2
        else:
            px, py = x1, y1
            near_d = d2
            far_d = d1

        # Score: prefere linhas longas que passam perto do centro
        score = far_d - near_d * 1.5
        if score > best_score and near_d < min_len * 0.6:
            best_score = score
            vx = px - cx
            vy = cy - py  # inverte Y
            best_angle = math.degrees(math.atan2(vx, vy)) % 360

    return best_angle


# ─────────────────────────────────────────────
# 4. ÂNGULO → DÍGITO
# ─────────────────────────────────────────────
def angulo_para_digito(angle, orientation):
    """
    Converte ângulo (0° = 12h, crescente horário) em dígito 0-9.
    Cada dígito ocupa 36°. O zero está no topo (12h = 0°).
    """
    if orientation == "anticlockwise":
        # Espelha o ângulo
        angle = (360 - angle) % 360

    # Cada divisão = 36°. O ponteiro entre dois dígitos arredonda para o menor.
    digito = int((angle + 18) % 360 // 36) % 10
    return digito


# ─────────────────────────────────────────────
# 5. REGRA DE CALIBRAÇÃO ENTRE DIALS ADJACENTES
# ─────────────────────────────────────────────
def calibrar_digitos(digitos, angulos):
    """
    Se o ponteiro de um dial estiver entre dois dígitos (zona cinza = ±5°
    da borda da divisão), usa o dial seguinte para decidir qual dos dois é.
    Regra: se o próximo dial ainda não passou de 5, arredonda para baixo.
    """
    resultado = list(digitos)

    for i in range(len(digitos) - 1):
        ang = angulos[i]
        if ang is None:
            continue

        orient = ORIENTACOES[i]
        if orient == "anticlockwise":
            ang_norm = (360 - ang) % 360
        else:
            ang_norm = ang

        # Posição dentro da divisão de 36° (0 a 35)
        pos_no_setor = ang_norm % 36

        # Zona ambígua: últimos 5° ou primeiros 5° de um setor
        if pos_no_setor >= 31 or pos_no_setor <= 5:
            proximo = digitos[i + 1] if i + 1 < len(digitos) else None
            if proximo is not None:
                # Se próximo < 5, o atual ainda não completou o dígito seguinte
                if pos_no_setor >= 31:
                    if proximo < 5:
                        resultado[i] = digitos[i]  # mantém atual
                    else:
                        resultado[i] = (digitos[i] + 1) % 10
                else:  # pos <= 5
                    if proximo >= 5:
                        resultado[i] = digitos[i]
                    else:
                        resultado[i] = (digitos[i] - 1) % 10

    return resultado


# ─────────────────────────────────────────────
# 6. FUNÇÃO PRINCIPAL
# ─────────────────────────────────────────────
def ler_medidor(img_bytes):
    img = load_cv2_from_bytes(img_bytes)
    if img is None:
        return {"erro": "Imagem inválida ou corrompida."}

    # Redimensiona se muito grande (melhora performance)
    h, w = img.shape[:2]
    if w > 1600:
        scale = 1600 / w
        img = cv2.resize(img, (1600, int(h * scale)))

    dials, centros = recortar_dials(img)

    if len(dials) < 4:
        return {"erro": "Não encontrei 4 dials na imagem. Tente uma foto mais próxima."}

    digitos_brutos = []
    angulos = []
    metodos = []

    for i, (dial_img, orient) in enumerate(zip(dials, ORIENTACOES)):
        angle, metodo = angulo_do_ponteiro(dial_img)
        angulos.append(round(angle, 1) if angle is not None else None)
        metodos.append(metodo)

        if angle is None:
            digitos_brutos.append(None)
        else:
            digitos_brutos.append(angulo_para_digito(angle, orient))

    if any(d is None for d in digitos_brutos):
        return {
            "erro": "Não consegui detectar todos os ponteiros. "
                    "Tente uma foto mais iluminada e sem reflexo.",
            "digitos": digitos_brutos,
            "angulos": angulos,
            "metodos": metodos,
        }

    # Aplica calibração entre dials adjacentes
    digitos_finais = calibrar_digitos(digitos_brutos, angulos)

    leitura = int("".join(str(d) for d in digitos_finais))

    return {
        "digitos": digitos_finais,
        "digitos_brutos": digitos_brutos,
        "angulos": angulos,
        "metodos": metodos,
        "leitura_kwh": leitura,
    }
