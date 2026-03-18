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
VISION_URL = "https://vision.googleapis.com/v1/images:annotate?key=" + GOOGLE_API_KEY

# Sentidos de cada dial (alternados)
ORIENTACOES = ["anticlockwise", "clockwise", "anticlockwise", "clockwise"]


# ─────────────────────────────────────────────
# CARREGAR IMAGEM
# ─────────────────────────────────────────────
def load_cv2_from_bytes(img_bytes):
    try:
        arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
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

        # Binariza (preto e branco) por região
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

    except Exception as e:
        h, w = img.shape[:2]
        gray = np.zeros((h, w), dtype=np.uint8)
        binary = np.zeros((h, w), dtype=np.uint8)
        return gray, binary


# ─────────────────────────────────────────────
# RECORTAR OS 4 DIALS
# ─────────────────────────────────────────────
def recortar_dials(img, gray):
    try:
        h, w = img.shape[:2]
        blur = cv2.GaussianBlur(gray, (9, 9), 2)
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

        dials = []
        centros = []

        if circles is not None:
            circles = np.round(circles[0]).astype(int)
            circles = sorted(circles, key=lambda c: c[0])

            filtrados = [circles[0]]
            for c in circles[1:]:
                if abs(c[0] - filtrados[-1][0]) > min_r * 1.5:
                    filtrados.append(c)
                if len(filtrados) == 4:
                    break

            if len(filtrados) == 4:
                for cx, cy, r in filtrados:
                    pad = int(r * 1.25)
                    x1 = max(cx - pad, 0)
                    y1 = max(cy - pad, 0)
                    x2 = min(cx + pad, w)
                    y2 = min(cy + pad, h)
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

    except Exception as e:
        # Fallback total: divide em 4 partes iguais
        h, w = img.shape[:2]
        dials = []
        centros = []
        dw = w // 4
        for i in range(4):
            x1 = i * dw
            x2 = (i + 1) * dw if i < 3 else w
            dials.append(img[0:h, x1:x2])
            centros.append(((x1 + x2) // 2, h // 2, min(dw, h) // 2))
        return dials, centros


# ─────────────────────────────────────────────
# GOOGLE VISION: detectar números no dial
# ─────────────────────────────────────────────
def detectar_numeros_no_dial(dial_img):
    """
    Envia o recorte do dial para o Google Vision e extrai
    os dígitos de 0-9 encontrados com suas posições (x, y).
    """
    if not GOOGLE_API_KEY:
        return []

    try:
        _, buf = cv2.imencode('.jpg', dial_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        b64 = base64.b64encode(buf.tobytes()).decode('utf-8')

        payload = {
            "requests": [{
                "image": {"content": b64},
                "features": [{"type": "TEXT_DETECTION"}]
            }]
        }

        resp = requests.post(VISION_URL, json=payload, timeout=10)
        data = resp.json()

        anotacoes = data.get("responses", [{}])[0].get("textAnnotations", [])
        if not anotacoes:
            return []

        numeros = []

        # Pula o primeiro item (é o texto completo), pega os individuais
        for item in anotacoes[1:]:
            texto = item.get("description", "").strip()
            if not re.match(r'^\d$', texto):
                continue

            vertices = item.get("boundingPoly", {}).get("vertices", [])
            if not vertices:
                continue

            xs = [v.get("x", 0) for v in vertices]
            ys = [v.get("y", 0) for v in vertices]
            px = sum(xs) // len(xs)
            py = sum(ys) // len(ys)

            numeros.append((int(texto), px, py))

        return numeros

    except Exception:
        return []


# ─────────────────────────────────────────────
# PIXEL → ÂNGULO em relação ao centro do dial
# ─────────────────────────────────────────────
def pixel_para_angulo(px, py, cx, cy):
    """
    0° = topo (12h), cresce no sentido horário.
    """
    dx = px - cx
    dy = cy - py  # inverte Y (Y cresce para baixo na imagem)
    angulo = math.degrees(math.atan2(dx, dy)) % 360
    return angulo


# ─────────────────────────────────────────────
# DETECTAR COR VERMELHA DO PONTEIRO
# ─────────────────────────────────────────────
def detectar_vermelho(dial_img, cx, cy, r):
    try:
        hsv = cv2.cvtColor(dial_img, cv2.COLOR_BGR2HSV)
        m1 = cv2.inRange(hsv, np.array([0, 120, 80]), np.array([8, 255, 255]))
        m2 = cv2.inRange(hsv, np.array([168, 120, 80]), np.array([180, 255, 255]))
        mask = cv2.bitwise_or(m1, m2)

        # Remove área central (pivot)
        cv2.circle(mask, (cx, cy), int(r * 0.14), 0, -1)

        pts = cv2.findNonZero(mask)
        if pts is None or len(pts) < 8:
            return None

        pts = pts.reshape(-1, 2).astype(np.float32)
        dists = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
        threshold = np.percentile(dists, 70)
        far = pts[dists >= threshold]

        if len(far) < 3:
            return None

        mx = float(np.mean(far[:, 0])) - cx
        my = cy - float(np.mean(far[:, 1]))
        return math.degrees(math.atan2(mx, my)) % 360

    except Exception:
        return None


# ─────────────────────────────────────────────
# DETECTAR PONTEIRO VIA HOUGH LINES (imagem P&B)
# ─────────────────────────────────────────────
def detectar_por_linhas(binary, cx, cy, r):
    try:
        min_len = int(r * 0.32)
        lines = cv2.HoughLinesP(
            binary, 1, np.pi / 360,
            threshold=10,
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

            if near_d > r * 0.42:
                continue

            comprimento = math.hypot(x2 - x1, y2 - y1)
            score = far_d * comprimento / max(near_d + 1, 1)
            ang = math.degrees(math.atan2(px - cx, cy - py)) % 360
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
        h, w = binary.shape[:2]
        contagens = []

        for ang_deg in range(0, 360, 5):
            ang_rad = math.radians(ang_deg)
            mascara = np.zeros((h, w), dtype=np.uint8)
            p1 = (cx, cy)
            a1 = ang_rad - math.radians(6)
            a2 = ang_rad + math.radians(6)
            p2 = (
                int(cx + r * 0.88 * math.sin(a1)),
                int(cy - r * 0.88 * math.cos(a1))
            )
            p3 = (
                int(cx + r * 0.88 * math.sin(a2)),
                int(cy - r * 0.88 * math.cos(a2))
            )
            cv2.fillPoly(mascara, [np.array([p1, p2, p3])], 255)
            overlap = cv2.bitwise_and(binary, mascara)
            contagens.append(cv2.countNonZero(overlap))

        melhor = int(np.argmax(contagens)) * 5
        return float(melhor)

    except Exception:
        return None


# ─────────────────────────────────────────────
# DETECTAR ÂNGULO DO PONTEIRO (junta tudo)
# ─────────────────────────────────────────────
def detectar_angulo_ponteiro(dial_img):
    try:
        h, w = dial_img.shape[:2]
        cx, cy = w // 2, h // 2
        r = min(w, h) // 2

        # Pré-processa o dial em P&B
        dial_gray = cv2.cvtColor(dial_img, cv2.COLOR_BGR2GRAY)
        dial_gray = cv2.bilateralFilter(dial_gray, 9, 75, 75)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
        dial_gray = clahe.apply(dial_gray)

        binary = cv2.adaptiveThreshold(
            dial_gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=21,
            C=8
        )

        # Máscara circular (ignora bordas do mostrador)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (cx, cy), int(r * 0.82), 255, -1)
        # Remove o pivot central
        cv2.circle(mask, (cx, cy), int(r * 0.14), 0, -1)
        binary = cv2.bitwise_and(binary, mask)

        # Tentativa 1: ponteiro vermelho
        ang = detectar_vermelho(dial_img, cx, cy, r)
        if ang is not None:
            return ang

        # Tentativa 2: Hough Lines na imagem P&B
        ang = detectar_por_linhas(binary, cx, cy, r)
        if ang is not None:
            return ang

        # Tentativa 3: varredura por fatias
        ang = detectar_por_fatias(binary, cx, cy, r)
        return ang

    except Exception:
        return None


# ─────────────────────────────────────────────
# REGRA PRINCIPAL: ponteiro + números impressos
# ─────────────────────────────────────────────
def digito_pelo_ponteiro_e_numeros(angulo_ponteiro, numeros_detectados, orientacao, dial_img):
    """
    Regras (como um leiturista humano):
    1. Converte posição de cada número impresso em ângulo.
    2. Ordena os números pelo ângulo no mostrador.
    3. Acha entre quais dois números o ponteiro está.
    4. Se entre dois números → dígito é o MENOR (o que já passou).
    5. Se exatamente em cima de um número → é aquele número.
    6. Se ponteiro tampando um número → deduz pelo anterior e próximo.
    """
    try:
        h, w = dial_img.shape[:2]
        cx = w // 2
        cy = h // 2

        if not numeros_detectados:
            return angulo_para_digito_simples(angulo_ponteiro, orientacao)

        # Converte cada número impresso para ângulo
        num_angulos = []
        for (digito, px, py) in numeros_detectados:
            ang = pixel_para_angulo(px, py, cx, cy)
            num_angulos.append((digito, ang))

        # Ordena por ângulo (sentido horário a partir do topo)
        num_angulos.sort(key=lambda x: x[1])

        # Ajusta o ângulo do ponteiro conforme orientação do dial
        ang_pont = angulo_ponteiro
        if orientacao == "anticlockwise":
            ang_pont = (360 - ang_pont) % 360

        # Verifica se o ponteiro está exatamente em cima de algum número
        TOLERANCIA_EXATA = 15  # graus

        for (digito, ang_num) in num_angulos:
            diff = abs(ang_pont - ang_num)
            if diff > 180:
                diff = 360 - diff
            if diff <= TOLERANCIA_EXATA:
                return digito

        # Encontra entre quais dois números o ponteiro está
        # O dígito é sempre o MENOR (o que já passou)
        for i in range(len(num_angulos)):
            d_atual, ang_atual = num_angulos[i]
            d_proximo, ang_proximo = num_angulos[(i + 1) % len(num_angulos)]

            # Caso normal
            if ang_atual <= ang_pont < ang_proximo:
                return d_atual

            # Caso de virada (entre 9 e 0, passando 360°)
            if ang_atual > ang_proximo:
                if ang_pont >= ang_atual or ang_pont < ang_proximo:
                    return d_atual

        # Fallback: número com ângulo mais próximo do ponteiro
        mais_proximo = min(
            num_angulos,
            key=lambda x: min(abs(ang_pont - x[1]), 360 - abs(ang_pont - x[1]))
        )
        return mais_proximo[0]

    except Exception:
        return angulo_para_digito_simples(angulo_ponteiro, orientacao)


# ─────────────────────────────────────────────
# FALLBACK: ângulo → dígito sem números visuais
# ─────────────────────────────────────────────
def angulo_para_digito_simples(angle, orientation):
    try:
        if orientation == "anticlockwise":
            angle = (360 - angle) % 360
        return int((angle + 18) % 360 // 36) % 10
    except Exception:
        return 0


# ─────────────────────────────────────────────
# DEDUZIR NÚMERO FALTANTE (ponteiro tampando)
# ─────────────────────────────────────────────
def deduzir_numero_faltante(numeros_detectados):
    """
    Se detectou 9 números (faltando 1), descobre qual é pela sequência 0-9.
    Exemplo: detectou 7, 9 mas não o 8 → o 8 está faltando (ponteiro em cima).
    """
    try:
        if len(numeros_detectados) != 9:
            return None

        detectados = set(d for (d, px, py) in numeros_detectados)
        todos = set(range(10))
        faltando = todos - detectados

        if len(faltando) == 1:
            return faltando.pop()
        return None

    except Exception:
        return None


# ─────────────────────────────────────────────
# FUNÇÃO PRINCIPAL
# ─────────────────────────────────────────────
def ler_medidor(img_bytes):
    try:
        img = load_cv2_from_bytes(img_bytes)
        if img is None:
            return {"erro": "Imagem inválida ou corrompida."}

        # Redimensiona se muito grande
        h, w = img.shape[:2]
        if w > 1400:
            scale = 1400 / w
            img = cv2.resize(img, (1400, int(h * scale)))

        # Pré-processamento geral
        gray, binary = preprocessar(img)

        # Recorta os 4 dials
        dials, centros = recortar_dials(img, gray)

        if len(dials) < 4:
            return {
                "erro": "Não encontrei 4 dials. Tente foto mais próxima e centralizada.",
                "dica": "Enquadre apenas os 4 relógios do medidor na foto."
            }

        digitos_finais = []
        angulos = []
        infos = []

        for i, (dial_img, orientacao) in enumerate(zip(dials, ORIENTACOES)):
            info_dial = f"dial {i + 1} ({orientacao}): "

            # 1. Detecta o ângulo do ponteiro
            angulo = detectar_angulo_ponteiro(dial_img)
            angulos.append(round(angulo, 1) if angulo is not None else None)

            if angulo is None:
                digitos_finais.append(None)
                infos.append(info_dial + "ponteiro não detectado")
                continue

            # 2. Detecta os números impressos no dial via Google Vision
            numeros = detectar_numeros_no_dial(dial_img)
            info_dial += f"{len(numeros)} números detectados"

            # 3. Verifica se falta algum número (ponteiro tampando)
            faltando = deduzir_numero_faltante(numeros)
            if faltando is not None:
                dh, dw = dial_img.shape[:2]
                cx = dw // 2
                cy = dh // 2
                r = min(dw, dh) // 2
                # Insere o número faltante na posição do ponteiro
                dx = int(cx + r * 0.38 * math.sin(math.radians(angulo)))
                dy = int(cy - r * 0.38 * math.cos(math.radians(angulo)))
                numeros.append((faltando, dx, dy))
                info_dial += f" | número {faltando} deduzido (ponteiro em cima)"

            infos.append(info_dial)

            # 4. Determina o dígito pela regra: ponteiro + números impressos
            digito = digito_pelo_ponteiro_e_numeros(
                angulo, numeros, orientacao, dial_img
            )
            digitos_finais.append(digito)

        # Verifica se leu todos os 4 dials
        if any(d is None for d in digitos_finais):
            return {
                "erro": "Não detectei todos os ponteiros. "
                        "Tente foto mais iluminada e sem reflexo no vidro.",
                "digitos": digitos_finais,
                "angulos": angulos,
                "infos": infos
            }

        leitura = int("".join(str(d) for d in digitos_finais))

        return {
            "digitos": digitos_finais,
            "angulos": angulos,
            "leitura_kwh": leitura,
            "infos": infos
        }

    except Exception as e:
        return {"erro": f"Erro interno no servidor: {str(e)}"}
