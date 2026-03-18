import cv2
import numpy as np
import math
import requests
import base64
import os
import re

# ─────────────────────────────────────────────
# CONFIGURAÇÕES
# ─────────────────────────────────────────────
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
VISION_URL     = "https://vision.googleapis.com/v1/images:annotate?key=" + GOOGLE_API_KEY
ORIENTACOES    = ["anticlockwise", "clockwise", "anticlockwise", "clockwise"]


# ─────────────────────────────────────────────
# CARREGAR IMAGEM
# ─────────────────────────────────────────────
def load_cv2_from_bytes(img_bytes):
    try:
        arr = np.frombuffer(img_bytes, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


# ─────────────────────────────────────────────
# PRÉ-PROCESSAMENTO P&B
# ─────────────────────────────────────────────
def preprocessar(img):
    try:
        # Converte para escala de cinza
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Remove reflexo preservando bordas
        gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

        # Aumenta contraste local
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
        gray = clahe.apply(gray)

        # Binariza por região (preto e branco)
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=21,
            C=8
        )

        # Remove ruído
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        return gray, binary

    except Exception:
        h, w = img.shape[:2]
        z = np.zeros((h, w), dtype=np.uint8)
        return z, z


# ─────────────────────────────────────────────
# RECORTAR OS 4 DIALS
# ─────────────────────────────────────────────
def recortar_dials(img, gray):
    try:
        h, w  = img.shape[:2]
        blur  = cv2.GaussianBlur(gray, (9, 9), 2)
        min_r = int(min(w, h) * 0.07)
        max_r = int(min(w, h) * 0.30)

        circles = cv2.HoughCircles(
            blur,
            cv2.HOUGH_GRADIENT,
            dp=1.0,
            minDist=int(min(w, h) * 0.14),
            param1=50,
            param2=28,
            minRadius=min_r,
            maxRadius=max_r
        )

        dials   = []
        centros = []

        if circles is not None:
            circles   = np.round(circles[0]).astype(int)
            circles   = sorted(circles, key=lambda c: c[0])
            filtrados = [circles[0]]

            for c in circles[1:]:
                if abs(c[0] - filtrados[-1][0]) > min_r * 1.5:
                    filtrados.append(c)
                if len(filtrados) == 4:
                    break

            if len(filtrados) == 4:
                for cx, cy, r in filtrados:
                    pad = int(r * 1.25)
                    x1  = max(cx - pad, 0)
                    y1  = max(cy - pad, 0)
                    x2  = min(cx + pad, w)
                    y2  = min(cy + pad, h)
                    dials.append(img[y1:y2, x1:x2])
                    centros.append((cx, cy, r))
                return dials, centros

        # Fallback: divide em 4 faixas iguais
        dw = w // 4
        for i in range(4):
            x1 = i * dw
            x2 = (i + 1) * dw if i < 3 else w
            dials.append(img[0:h, x1:x2])
            centros.append(((x1 + x2) // 2, h // 2, min(dw, h) // 2))
        return dials, centros

    except Exception:
        h, w = img.shape[:2]
        dw   = w // 4
        dials   = []
        centros = []
        for i in range(4):
            x1 = i * dw
            x2 = (i + 1) * dw if i < 3 else w
            dials.append(img[0:h, x1:x2])
            centros.append(((x1 + x2) // 2, h // 2, min(dw, h) // 2))
        return dials, centros


# ─────────────────────────────────────────────
# GOOGLE VISION: detectar números no dial
# ─────────────────────────────────────────────
def detectar_numeros_no_dial(dial_gray):
    if not GOOGLE_API_KEY:
        return []
    try:
        dial_bgr = cv2.cvtColor(dial_gray, cv2.COLOR_GRAY2BGR)
        _, buf   = cv2.imencode('.jpg', dial_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        b64      = base64.b64encode(buf.tobytes()).decode('utf-8')

        payload = {
            "requests": [{
                "image":    {"content": b64},
                "features": [{"type": "TEXT_DETECTION"}]
            }]
        }

        resp  = requests.post(VISION_URL, json=payload, timeout=10)
        data  = resp.json()
        anots = data.get("responses", [{}])[0].get("textAnnotations", [])

        if not anots:
            return []

        numeros = []
        for item in anots[1:]:
            texto = item.get("description", "").strip()
            if not re.match(r'^\d$', texto):
                continue
            verts = item.get("boundingPoly", {}).get("vertices", [])
            if not verts:
                continue
            xs = [v.get("x", 0) for v in verts]
            ys = [v.get("y", 0) for v in verts]
            numeros.append((int(texto), sum(xs) // len(xs), sum(ys) // len(ys)))

        return numeros

    except Exception:
        return []


# ─────────────────────────────────────────────
# PIXEL → ÂNGULO em relação ao centro
# ─────────────────────────────────────────────
def pixel_para_angulo(px, py, cx, cy):
    # 0° = topo (12h), cresce sentido horário
    return math.degrees(math.atan2(px - cx, cy - py)) % 360


# ─────────────────────────────────────────────
# DETECTAR PONTEIRO VERMELHO
# ─────────────────────────────────────────────
def detectar_vermelho(dial_color, cx, cy, r):
    try:
        hsv = cv2.cvtColor(dial_color, cv2.COLOR_BGR2HSV)
        m1  = cv2.inRange(hsv, np.array([0,   120, 80]), np.array([8,   255, 255]))
        m2  = cv2.inRange(hsv, np.array([168, 120, 80]), np.array([180, 255, 255]))
        mask = cv2.bitwise_or(m1, m2)

        # Remove pivot central
        cv2.circle(mask, (cx, cy), int(r * 0.14), 0, -1)

        pts = cv2.findNonZero(mask)
        if pts is None or len(pts) < 8:
            return None

        pts   = pts.reshape(-1, 2).astype(np.float32)
        dists = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
        far   = pts[dists >= np.percentile(dists, 70)]

        if len(far) < 3:
            return None

        mx = float(np.mean(far[:, 0])) - cx
        my = cy - float(np.mean(far[:, 1]))
        return math.degrees(math.atan2(mx, my)) % 360

    except Exception:
        return None


# ─────────────────────────────────────────────
# DETECTAR PONTEIRO VIA HOUGH LINES
# ─────────────────────────────────────────────
def detectar_por_linhas(binary, cx, cy, r):
    try:
        lines = cv2.HoughLinesP(
            binary, 1, np.pi / 360,
            threshold=10,
            minLineLength=int(r * 0.32),
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
                px, py   = x2, y2
                near_d   = d1
                far_d    = d2
            else:
                px, py   = x1, y1
                near_d   = d2
                far_d    = d1

            if near_d > r * 0.42:
                continue

            comprimento = math.hypot(x2 - x1, y2 - y1)
            score = far_d * comprimento / max(near_d + 1, 1)
            ang   = math.degrees(math.atan2(px - cx, cy - py)) % 360
            candidatos.append((score, ang))

        if not candidatos:
            return None

        candidatos.sort(reverse=True)
        return candidatos[0][1]

    except Exception:
        return None


# ─────────────────────────────────────────────
# DETECTAR PONTEIRO POR VARREDURA DE FATIAS
# ─────────────────────────────────────────────
def detectar_por_fatias(binary, cx, cy, r):
    try:
        h, w      = binary.shape[:2]
        contagens = []

        for ang_deg in range(0, 360, 5):
            ar  = math.radians(ang_deg)
            msk = np.zeros((h, w), dtype=np.uint8)
            p1  = (cx, cy)
            p2  = (
                int(cx + r * 0.88 * math.sin(ar - math.radians(6))),
                int(cy - r * 0.88 * math.cos(ar - math.radians(6)))
            )
            p3  = (
                int(cx + r * 0.88 * math.sin(ar + math.radians(6))),
                int(cy - r * 0.88 * math.cos(ar + math.radians(6)))
            )
            cv2.fillPoly(msk, [np.array([p1, p2, p3])], 255)
            contagens.append(cv2.countNonZero(cv2.bitwise_and(binary, msk)))

        return float(int(np.argmax(contagens)) * 5)

    except Exception:
        return None


# ─────────────────────────────────────────────
# DETECTAR ÂNGULO DO PONTEIRO
# ─────────────────────────────────────────────
def detectar_angulo_ponteiro(dial_gray):
    try:
        h, w   = dial_gray.shape[:2]
        cx, cy = w // 2, h // 2
        r      = min(w, h) // 2

        # Máscara circular
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (cx, cy), int(r * 0.82), 255, -1)
        cv2.circle(mask, (cx, cy), int(r * 0.14),   0, -1)

        # Binariza
        binary = cv2.adaptiveThreshold(
            dial_gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=21,
            C=8
        )
        binary = cv2.bitwise_and(binary, mask)

        # Tentativa 1: Hough Lines
        ang = detectar_por_linhas(binary, cx, cy, r)
        if ang is not None:
            return ang

        # Tentativa 2: varredura por fatias
        ang = detectar_por_fatias(binary, cx, cy, r)
        return ang

    except Exception:
        return None


# ─────────────────────────────────────────────
# DEDUZIR NÚMERO FALTANTE (ponteiro tampando)
# ─────────────────────────────────────────────
def deduzir_numero_faltante(numeros):
    try:
        if len(numeros) != 9:
            return None
        detectados = set(d for (d, px, py) in numeros)
        faltando   = set(range(10)) - detectados
        return faltando.pop() if len(faltando) == 1 else None
    except Exception:
        return None


# ─────────────────────────────────────────────
# REGRA: ponteiro + números impressos → dígito
# Regras como um leiturista humano:
# 1. Se exatamente em cima de um número → é esse número
# 2. Se entre dois números → é sempre o MENOR (o que já passou)
# 3. Se um número está faltando (ponteiro em cima) → deduz pelo anterior e próximo
# ─────────────────────────────────────────────
def digito_pelo_ponteiro_e_numeros(angulo, numeros, orientacao, cx, cy):
    try:
        if not numeros:
            return angulo_simples(angulo, orientacao)

        # Converte cada número para ângulo no mostrador
        num_ang = []
        for (d, px, py) in numeros:
            a = pixel_para_angulo(px, py, cx, cy)
            num_ang.append((d, a))

        # Ordena por ângulo
        num_ang.sort(key=lambda x: x[1])

        # Ajusta ângulo do ponteiro conforme sentido do dial
        ang = (360 - angulo) % 360 if orientacao == "anticlockwise" else angulo

        # REGRA 1: exatamente em cima de um número (±15°)
        for d, a in num_ang:
            diff = abs(ang - a)
            if diff > 180:
                diff = 360 - diff
            if diff <= 15:
                return d

        # REGRA 2: entre dois números → pega sempre o MENOR
        for i in range(len(num_ang)):
            d_cur,  a_cur  = num_ang[i]
            d_next, a_next = num_ang[(i + 1) % len(num_ang)]

            # Caso normal
            if a_cur <= ang < a_next:
                return d_cur

            # Caso de virada (9 → 0, passando pelo 360°)
            if a_cur > a_next:
                if ang >= a_cur or ang < a_next:
                    return d_cur

        # Fallback: número com ângulo mais próximo
        return min(num_ang,
                   key=lambda x: min(abs(ang - x[1]), 360 - abs(ang - x[1])))[0]

    except Exception:
        return angulo_simples(angulo, orientacao)


# ─────────────────────────────────────────────
# FALLBACK: ângulo → dígito sem números visuais
# ─────────────────────────────────────────────
def angulo_simples(angle, orientation):
    try:
        if orientation == "anticlockwise":
            angle = (360 - angle) % 360
        return int((angle + 18) % 360 // 36) % 10
    except Exception:
        return 0


# ─────────────────────────────────────────────
# LER UM DIAL INDIVIDUAL
# chamado pelo main.py para cada dial separado
# ─────────────────────────────────────────────
def ler_dial_individual(dial_gray, orientacao, indice):
    try:
        h, w   = dial_gray.shape[:2]
        cx, cy = w // 2, h // 2
        r      = min(w, h) // 2

        # 1. Detecta ângulo do ponteiro
        angulo = detectar_angulo_ponteiro(dial_gray)

        if angulo is None:
            return {
                "digito": None,
                "angulo": None,
                "info":   f"dial {indice+1}: ponteiro não detectado"
            }

        # 2. Detecta números impressos via Google Vision
        numeros = detectar_numeros_no_dial(dial_gray)
        info    = f"dial {indice+1}: {len(numeros)} núm, ang={round(angulo,1)}°"

        # 3. Número faltante — ponteiro está em cima dele
        faltando = deduzir_numero_faltante(numeros)
        if faltando is not None:
            dx = int(cx + r * 0.38 * math.sin(math.radians(angulo)))
            dy = int(cy - r * 0.38 * math.cos(math.radians(angulo)))
            numeros.append((faltando, dx, dy))
            info += f" | nº {faltando} deduzido"

        # 4. Determina dígito pela regra ponteiro + números
        digito = digito_pelo_ponteiro_e_numeros(angulo, numeros, orientacao, cx, cy)

        return {
            "digito": digito,
            "angulo": round(angulo, 1),
            "info":   info
        }

    except Exception as e:
        return {
            "digito": None,
            "angulo": None,
            "info":   f"dial {indice+1}: erro - {str(e)}"
        }


# ─────────────────────────────────────────────
# FUNÇÃO PRINCIPAL (compatibilidade com versão antiga)
# ─────────────────────────────────────────────
def ler_medidor(img_bytes):
    try:
        img = load_cv2_from_bytes(img_bytes)
        if img is None:
            return {"erro": "Imagem inválida ou corrompida."}

        h, w = img.shape[:2]
        if w > 1400:
            scale = 1400 / w
            img   = cv2.resize(img, (1400, int(h * scale)))

        gray, _ = preprocessar(img)
        dials, centros = recortar_dials(img, gray)

        if len(dials) < 4:
            return {"erro": "Não encontrei 4 dials."}

        digitos = []
        angulos = []
        infos   = []

        for i, (dial_img, orientacao) in enumerate(zip(dials, ORIENTACOES)):
            dial_gray = cv2.cvtColor(dial_img, cv2.COLOR_BGR2GRAY)
            dial_gray = cv2.bilateralFilter(dial_gray, 9, 75, 75)
            clahe     = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
            dial_gray = clahe.apply(dial_gray)

            resultado = ler_dial_individual(dial_gray, orientacao, i)
            digitos.append(resultado.get("digito"))
            angulos.append(resultado.get("angulo"))
            infos.append(resultado.get("info"))

        if any(d is None for d in digitos):
            return {
                "erro":    "Não detectei todos os ponteiros.",
                "digitos": digitos,
                "angulos": angulos,
                "infos":   infos
            }

        leitura = int("".join(str(d) for d in digitos))
        return {
            "digitos":     digitos,
            "angulos":     angulos,
            "leitura_kwh": leitura,
            "infos":       infos
        }

    except Exception as e:
        return {"erro": f"Erro interno: {str(e)}"}
