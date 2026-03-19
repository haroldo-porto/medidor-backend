import cv2
import numpy as np
import math
import requests
import base64
import os
import re

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
VISION_URL     = "https://vision.googleapis.com/v1/images:annotate?key=" + GOOGLE_API_KEY
ORIENTACOES    = ["anticlockwise", "clockwise", "anticlockwise", "clockwise"]


# ─────────────────────────────────────────────
# PRÉ-PROCESSAMENTO (só para detecção interna)
# ─────────────────────────────────────────────
def preprocessar(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
        gray = clahe.apply(gray)
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=21, C=8
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        return gray, binary
    except Exception:
        h, w = img.shape[:2]
        z = np.zeros((h, w), dtype=np.uint8)
        return z, z


# ─────────────────────────────────────────────
# RECORTAR OS 4 DIALS
# Clusteriza por coluna X, escolhe 1 por coluna
# ─────────────────────────────────────────────
def recortar_dials(img, gray):
    try:
        h, w = img.shape[:2]
        blur  = cv2.GaussianBlur(gray, (9, 9), 2)
        min_r = int(min(w, h) * 0.06)
        max_r = int(min(w, h) * 0.20)

        circles = cv2.HoughCircles(
            blur,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=int(min(w, h) * 0.08),
            param1=80,
            param2=25,
            minRadius=min_r,
            maxRadius=max_r
        )

        if circles is None:
            raise ValueError("Nenhum círculo encontrado")

        circles = np.round(circles[0]).astype(int)

        # Filtra pela faixa vertical onde ficam os dials
        ys    = [c[1] for c in circles]
        y_med = float(np.median(ys))
        y_tol = int(h * 0.20)
        circles = [c for c in circles if abs(c[1] - y_med) <= y_tol]

        if len(circles) < 4:
            raise ValueError("Menos de 4 círculos na faixa horizontal")

        # Ordena por X
        circles = sorted(circles, key=lambda c: c[0])

        # Raio mediano para limiar de agrupamento
        raios = [c[2] for c in circles]
        r_med = float(np.median(raios))
        if r_med <= 0:
            r_med = min_r

        # Agrupa em colunas por proximidade de X
        colunas = []
        for c in circles:
            cx = c[0]
            colocado = False
            for coluna in colunas:
                cx_col = float(np.mean([cc[0] for cc in coluna]))
                if abs(cx - cx_col) <= r_med * 1.2:
                    coluna.append(c)
                    colocado = True
                    break
            if not colocado:
                colunas.append([c])

        # Funde colunas mais próximas até ter 4
        while len(colunas) > 4:
            melhor_i, melhor_j, melhor_dist = 0, 1, 1e9
            for i in range(len(colunas)):
                for j in range(i + 1, len(colunas)):
                    cx_i = float(np.mean([c[0] for c in colunas[i]]))
                    cx_j = float(np.mean([c[0] for c in colunas[j]]))
                    dist = abs(cx_i - cx_j)
                    if dist < melhor_dist:
                        melhor_dist = dist
                        melhor_i, melhor_j = i, j
            colunas[melhor_i].extend(colunas[melhor_j])
            del colunas[melhor_j]

        if len(colunas) < 4:
            raise ValueError("Menos de 4 colunas")

        # Em cada coluna escolhe 1 círculo (mais central em Y)
        escolhidos = []
        for coluna in colunas:
            ys_col  = [c[1] for c in coluna]
            y_med_c = float(np.median(ys_col))
            melhor  = min(coluna, key=lambda c: abs(c[1] - y_med_c))
            escolhidos.append(melhor)

        # Ordena os 4 finais por X
        escolhidos = sorted(escolhidos, key=lambda c: c[0])

        # Recorte quadrado pequeno (1 dial por imagem)
        dials   = []
        centros = []
        for cx, cy, r in escolhidos:
            lado = int(r * 1.4)
            half = lado // 2
            x1 = max(cx - half, 0)
            x2 = min(cx + half, w)
            y1 = max(cy - half, 0)
            y2 = min(cy + half, h)
            dials.append(img[y1:y2, x1:x2])
            centros.append((cx, cy, r))

        if len(dials) != 4:
            raise ValueError("Não formou 4 recortes")

        return dials, centros

    except Exception:
        # Fallback: 4 faixas verticais iguais
        h, w    = img.shape[:2]
        dials   = []
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
def detectar_numeros_no_dial(dial_gray):
    if not GOOGLE_API_KEY:
        return []
    try:
        dial_bgr = cv2.cvtColor(dial_gray, cv2.COLOR_GRAY2BGR)
        _, buf   = cv2.imencode('.jpg', dial_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        b64      = base64.b64encode(buf.tobytes()).decode('utf-8')

        payload = {"requests": [{
            "image":    {"content": b64},
            "features": [{"type": "TEXT_DETECTION"}]
        }]}

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


def pixel_para_angulo(px, py, cx, cy):
    return math.degrees(math.atan2(px - cx, cy - py)) % 360


# ─────────────────────────────────────────────
# DETECTAR PONTEIRO VIA HOUGH LINES
# (usa binário interno, não afeta visual)
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
                px, py, near_d, far_d = x2, y2, d1, d2
            else:
                px, py, near_d, far_d = x1, y1, d2, d1
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
# Usa binário INTERNO — a imagem exibida continua cinza suave
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

        # Binário INTERNO para detecção (não afeta visual)
        blur = cv2.GaussianBlur(dial_gray, (5, 5), 0)
        binary = cv2.adaptiveThreshold(
            blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=21, C=8
        )
        binary = cv2.bitwise_and(binary, mask)

        # Tenta Hough Lines primeiro
        ang = detectar_por_linhas(binary, cx, cy, r)
        if ang is not None:
            return ang

        # Fallback: varredura por fatias
        return detectar_por_fatias(binary, cx, cy, r)

    except Exception:
        return None


# ─────────────────────────────────────────────
# DEDUZIR NÚMERO FALTANTE
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
# DÍGITO PELO PONTEIRO + NÚMEROS
# ─────────────────────────────────────────────
def digito_pelo_ponteiro_e_numeros(angulo, numeros, orientacao, cx, cy):
    try:
        if not numeros:
            return angulo_simples(angulo, orientacao)

        num_ang = [(d, pixel_para_angulo(px, py, cx, cy)) for (d, px, py) in numeros]
        num_ang.sort(key=lambda x: x[1])

        ang = (360 - angulo) % 360 if orientacao == "anticlockwise" else angulo

        # Exatamente em cima (±15°)
        for d, a in num_ang:
            diff = abs(ang - a)
            if diff > 180: diff = 360 - diff
            if diff <= 15:
                return d

        # Entre dois números → pega o menor (o que já passou)
        for i in range(len(num_ang)):
            d_cur,  a_cur  = num_ang[i]
            d_next, a_next = num_ang[(i + 1) % len(num_ang)]
            if a_cur <= ang < a_next:
                return d_cur
            if a_cur > a_next and (ang >= a_cur or ang < a_next):
                return d_cur

        return min(num_ang,
                   key=lambda x: min(abs(ang - x[1]), 360 - abs(ang - x[1])))[0]
    except Exception:
        return angulo_simples(angulo, orientacao)


def angulo_simples(angle, orientation):
    try:
        if orientation == "anticlockwise":
            angle = (360 - angle) % 360
        return int((angle + 18) % 360 // 36) % 10
    except Exception:
        return 0


# ─────────────────────────────────────────────
# LER UM DIAL INDIVIDUAL
# ─────────────────────────────────────────────
def ler_dial_individual(dial_gray, orientacao, indice):
    try:
        h, w   = dial_gray.shape[:2]
        cx, cy = w // 2, h // 2
        r      = min(w, h) // 2

        # 1) Detecta ângulo do ponteiro (usa binário interno)
        angulo = detectar_angulo_ponteiro(dial_gray)
        if angulo is None:
            return {
                "digito": None,
                "angulo": None,
                "info":   f"dial {indice+1}: ponteiro não detectado"
            }

        # 2) Detecta números via Google Vision
        numeros = detectar_numeros_no_dial(dial_gray)
        info    = f"dial {indice+1}: {len(numeros)} núm, ang={round(angulo, 1)}°"

        # 3) Número faltante (ponteiro em cima)
        faltando = deduzir_numero_faltante(numeros)
        if faltando is not None:
            dx = int(cx + r * 0.38 * math.sin(math.radians(angulo)))
            dy = int(cy - r * 0.38 * math.cos(math.radians(angulo)))
            numeros.append((faltando, dx, dy))
            info += f" | nº {faltando} deduzido"

        # 4) Dígito final
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
# FUNÇÃO PRINCIPAL (compatibilidade)
# ─────────────────────────────────────────────
def ler_medidor(img_bytes):
    try:
        arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return {"erro": "Imagem inválida."}

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
            # Cinza suave para leitura
            dial_gray = cv2.cvtColor(dial_img, cv2.COLOR_BGR2GRAY)
            clahe     = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            dial_gray = clahe.apply(dial_gray)
            dial_gray = cv2.convertScaleAbs(dial_gray, alpha=1.2, beta=-10)

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
