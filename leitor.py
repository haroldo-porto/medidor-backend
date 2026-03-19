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
# PRÉ-PROCESSAMENTO
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
# Com correção automática de inclinação
# ─────────────────────────────────────────────
def recortar_dials(img, gray):
    try:
        h, w  = img.shape[:2]
        blur  = cv2.GaussianBlur(gray, (9, 9), 2)
        min_r = int(min(w, h) * 0.06)
        max_r = int(min(w, h) * 0.22)

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
            raise ValueError("Nenhum círculo")

        circles = np.round(circles[0]).astype(int)

        # Filtra pela faixa vertical dos dials
        ys    = [c[1] for c in circles]
        y_med = float(np.median(ys))
        y_tol = int(h * 0.20)
        circles = [c for c in circles if abs(c[1] - y_med) <= y_tol]

        if len(circles) < 4:
            raise ValueError("Menos de 4 círculos")

        circles = sorted(circles, key=lambda c: c[0])

        raios = [c[2] for c in circles]
        r_med = float(np.median(raios))
        if r_med <= 0:
            r_med = min_r

        # Agrupa em 4 colunas por X
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

        # Funde colunas até ter 4
        while len(colunas) > 4:
            mi, mj, md = 0, 1, 1e9
            for i in range(len(colunas)):
                for j in range(i + 1, len(colunas)):
                    cx_i = float(np.mean([c[0] for c in colunas[i]]))
                    cx_j = float(np.mean([c[0] for c in colunas[j]]))
                    d = abs(cx_i - cx_j)
                    if d < md:
                        md, mi, mj = d, i, j
            colunas[mi].extend(colunas[mj])
            del colunas[mj]

        if len(colunas) < 4:
            raise ValueError("Menos de 4 colunas")

        # Escolhe 1 círculo por coluna (mais central em Y)
        escolhidos = []
        for coluna in colunas:
            ys_c   = [c[1] for c in coluna]
            y_mc   = float(np.median(ys_c))
            melhor = min(coluna, key=lambda c: abs(c[1] - y_mc))
            escolhidos.append(melhor)

        escolhidos = sorted(escolhidos, key=lambda c: c[0])

        # ── CORREÇÃO DE INCLINAÇÃO ──
        xs_e = np.array([float(c[0]) for c in escolhidos])
        ys_e = np.array([float(c[1]) for c in escolhidos])
        coef = np.polyfit(xs_e, ys_e, 1)
        angle_deg = math.degrees(math.atan2(coef[0], 1.0))

        img_work  = img
        gray_work = gray

        if abs(angle_deg) > 0.5:
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, -angle_deg, 1.0)
            img_work  = cv2.warpAffine(img,  M, (w, h),
                                       flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_REFLECT)
            gray_work = cv2.warpAffine(gray, M, (w, h),
                                       flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_REFLECT)
            pts = np.array([[c[0], c[1]] for c in escolhidos],
                           dtype=np.float32).reshape(-1, 1, 2)
            pts_rot = cv2.transform(pts, M)
            escolhidos = [
                (int(pts_rot[i][0][0]), int(pts_rot[i][0][1]), escolhidos[i][2])
                for i in range(4)
            ]

        # Recorte refinado com detecção do anel
        dials   = []
        centros = []

        for cx, cy, r in escolhidos:
            pad  = int(r * 1.8)
            x1_i = max(cx - pad, 0)
            x2_i = min(cx + pad, w)
            y1_i = max(cy - pad, 0)
            y2_i = min(cy + pad, h)

            roi_gray = gray_work[y1_i:y2_i, x1_i:x2_i]
            rh, rw   = roi_gray.shape[:2]
            cx_r     = rw // 2
            cy_r     = rh // 2

            edges    = cv2.Canny(roi_gray, 50, 150)
            r_min    = int(r * 0.5)
            r_max    = int(r * 1.1)
            melhor_r = r
            melhor_v = -1

            for rr in range(r_min, r_max, max(1, (r_max - r_min) // 12)):
                msk = np.zeros_like(edges)
                cv2.circle(msk, (cx_r, cy_r), rr, 255, 2)
                val = cv2.countNonZero(cv2.bitwise_and(edges, msk))
                if val > melhor_v:
                    melhor_v = val
                    melhor_r = rr

            half = int(melhor_r * 1.6)
            x1   = max(cx - half, 0)
            x2   = min(cx + half, w)
            y1   = max(cy - half, 0)
            y2   = min(cy + half, h)

            crop = img_work[y1:y2, x1:x2]
            crop = cv2.resize(crop, (200, 200))
            dials.append(crop)
            centros.append((cx, cy, melhor_r))

        if len(dials) != 4:
            raise ValueError("Não formou 4 recortes")

        return dials, centros

    except Exception:
        h, w    = img.shape[:2]
        dials   = []
        centros = []
        dw = w // 4
        for i in range(4):
            x1 = i * dw
            x2 = (i + 1) * dw if i < 3 else w
            crop = img[0:h, x1:x2]
            crop = cv2.resize(crop, (200, 200))
            dials.append(crop)
            centros.append(((x1 + x2) // 2, h // 2, min(dw, h) // 2))
        return dials, centros


# ─────────────────────────────────────────────
# GOOGLE VISION — só dígitos únicos (0-9)
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

        h, w = dial_gray.shape[:2]
        cx   = w // 2
        cy   = h // 2
        r    = min(w, h) // 2

        numeros = []
        vistos  = set()

        for item in anots[1:]:
            texto = item.get("description", "").strip()

            # aceita SOMENTE dígito único 0-9
            if not re.match(r'^\d$', texto):
                continue

            verts = item.get("boundingPoly", {}).get("vertices", [])
            if not verts:
                continue

            xs  = [v.get("x", 0) for v in verts]
            ys  = [v.get("y", 0) for v in verts]
            px  = sum(xs) // len(xs)
            py  = sum(ys) // len(ys)

            # descarta detecções no centro (região do ponteiro, não de números)
            dist_centro = math.hypot(px - cx, py - cy)
            if dist_centro < r * 0.25:
                continue

            # descarta detecções fora do círculo do dial
            if dist_centro > r * 1.05:
                continue

            d = int(texto)

            # evita duplicatas do mesmo dígito muito próximas
            duplicata = False
            for (dd, ppx, ppy) in numeros:
                if dd == d and math.hypot(px - ppx, py - ppy) < r * 0.15:
                    duplicata = True
                    break
            if duplicata:
                continue

            numeros.append((d, px, py))

        return numeros

    except Exception:
        return []


# ─────────────────────────────────────────────
# CONVERTE PIXEL → ÂNGULO (0° = topo, sentido horário)
# ─────────────────────────────────────────────
def pixel_para_angulo(px, py, cx, cy):
    return math.degrees(math.atan2(px - cx, cy - py)) % 360


# ─────────────────────────────────────────────
# DETECTAR PONTEIRO — HoughLinesP
# ─────────────────────────────────────────────
def detectar_por_linhas(binary, cx, cy, r):
    try:
        lines = cv2.HoughLinesP(
            binary, 1, np.pi / 360,
            threshold=10,
            minLineLength=int(r * 0.30),
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
# DETECTAR PONTEIRO — varredura por fatias
# ─────────────────────────────────────────────
def detectar_por_fatias(binary, cx, cy, r):
    try:
        h, w      = binary.shape[:2]
        contagens = []
        for ang_deg in range(0, 360, 3):
            ar  = math.radians(ang_deg)
            msk = np.zeros((h, w), dtype=np.uint8)
            p1  = (cx, cy)
            p2  = (
                int(cx + r * 0.88 * math.sin(ar - math.radians(5))),
                int(cy - r * 0.88 * math.cos(ar - math.radians(5)))
            )
            p3  = (
                int(cx + r * 0.88 * math.sin(ar + math.radians(5))),
                int(cy - r * 0.88 * math.cos(ar + math.radians(5)))
            )
            cv2.fillPoly(msk, [np.array([p1, p2, p3])], 255)
            contagens.append(cv2.countNonZero(cv2.bitwise_and(binary, msk)))
        return float(int(np.argmax(contagens)) * 3)
    except Exception:
        return None


# ─────────────────────────────────────────────
# DETECTAR ÂNGULO DO PONTEIRO
# Usa binário INTERNO — visual continua cinza suave
# ─────────────────────────────────────────────
def detectar_angulo_ponteiro(dial_gray):
    try:
        h, w   = dial_gray.shape[:2]
        cx, cy = w // 2, h // 2
        r      = min(w, h) // 2

        # máscara: só analisa o anel entre 14% e 82% do raio
        # (centro = eixo do ponteiro; borda externa = números)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (cx, cy), int(r * 0.82), 255, -1)
        cv2.circle(mask, (cx, cy), int(r * 0.14),   0, -1)

        blur = cv2.GaussianBlur(dial_gray, (5, 5), 0)
        binary = cv2.adaptiveThreshold(
            blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=21, C=8
        )
        binary = cv2.bitwise_and(binary, mask)

        # tenta HoughLines primeiro (mais preciso)
        ang = detectar_por_linhas(binary, cx, cy, r)
        if ang is not None:
            return ang

        # fallback: varredura de fatias
        return detectar_por_fatias(binary, cx, cy, r)
    except Exception:
        return None


# ─────────────────────────────────────────────
# REGRA PRINCIPAL DE LEITURA DO DÍGITO
#
# Regra exata conforme o medidor:
#   "o dígito é o número imediatamente ANTES do ponteiro,
#    no sentido de rotação do dial.
#    Só usa o PRÓXIMO se o ponteiro estiver colado nele (< 5°)"
# ─────────────────────────────────────────────
def digito_pelo_ponteiro_e_numeros(angulo_ponteiro, numeros, orientacao, cx, cy):
    try:
        if not numeros:
            # sem números detectados → fallback geométrico simples
            return angulo_simples(angulo_ponteiro, orientacao)

        # converte cada número (pixel) para ângulo no círculo
        num_ang = []
        for (d, px, py) in numeros:
            a = pixel_para_angulo(px, py, cx, cy)
            num_ang.append((d, a))

        # ajusta o sentido de rotação
        if orientacao == "anticlockwise":
            # inverte tudo para trabalhar sempre no sentido horário
            ang_pont = (360 - angulo_ponteiro) % 360
            num_ang  = [(d, (360 - a) % 360) for (d, a) in num_ang]
        else:
            ang_pont = angulo_ponteiro

        # ordena pelos ângulos (0..360, sentido horário)
        num_ang.sort(key=lambda x: x[1])

        n = len(num_ang)

        # percorre todos os pares de números consecutivos
        # e acha entre quais o ponteiro está
        for i in range(n):
            d_antes, a_antes = num_ang[i]
            d_depois, a_depois = num_ang[(i + 1) % n]

            # intervalo pode cruzar o 0°/360°
            if a_antes <= a_depois:
                # intervalo normal [a_antes, a_depois)
                no_intervalo = a_antes <= ang_pont < a_depois
            else:
                # intervalo cruza 360°
                no_intervalo = ang_pont >= a_antes or ang_pont < a_depois

            if no_intervalo:
                # quantos graus falta para chegar no próximo número?
                if a_antes <= a_depois:
                    falta_para_proximo = a_depois - ang_pont
                else:
                    if ang_pont >= a_antes:
                        falta_para_proximo = (360 - ang_pont) + a_depois
                    else:
                        falta_para_proximo = a_depois - ang_pont

                # REGRA: devolve o número ANTERIOR
                # exceto se o ponteiro estiver praticamente em cima do próximo
                if falta_para_proximo < 5:
                    print(f"  → ponteiro colado no próximo: {d_depois} "
                          f"(falta {falta_para_proximo:.1f}°)")
                    return d_depois
                else:
                    print(f"  → entre {d_antes} e {d_depois}, "
                          f"falta {falta_para_proximo:.1f}° → dígito={d_antes}")
                    return d_antes

        # fallback: número cujo ângulo está mais próximo do ponteiro
        mais_proximo = min(
            num_ang,
            key=lambda x: min(abs(ang_pont - x[1]), 360 - abs(ang_pont - x[1]))
        )
        print(f"  → fallback mais próximo: {mais_proximo[0]}")
        return mais_proximo[0]

    except Exception as e:
        print(f"  → exceção em digito_pelo_ponteiro: {e}")
        return angulo_simples(angulo_ponteiro, orientacao)


# ─────────────────────────────────────────────
# FALLBACK GEOMÉTRICO SIMPLES
# Só usado quando Vision não detectou nenhum número
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
# ─────────────────────────────────────────────
def ler_dial_individual(dial_gray, orientacao, indice):
    try:
        h, w   = dial_gray.shape[:2]
        cx, cy = w // 2, h // 2
        r      = min(w, h) // 2

        # 1) Detecta ângulo do ponteiro
        angulo = detectar_angulo_ponteiro(dial_gray)

        print(f"\n[dial {indice+1}] orientacao={orientacao}")
        print(f"  angulo ponteiro = {angulo}")

        if angulo is None:
            print(f"  ERRO: ponteiro não detectado")
            return {
                "digito": None,
                "angulo": None,
                "info":   f"dial {indice+1}: ponteiro não detectado"
            }

        # 2) Detecta números via Google Vision
        numeros = detectar_numeros_no_dial(dial_gray)
        print(f"  numeros detectados = {numeros}")

        info = f"dial {indice+1}: {len(numeros)} núm, ang={round(angulo,1)}°"

        # 3) Dígito pela regra do ponteiro entre dois números
        digito = digito_pelo_ponteiro_e_numeros(
            angulo, numeros, orientacao, cx, cy
        )

        print(f"  DIGITO FINAL = {digito}")

        return {
            "digito": digito,
            "angulo": round(angulo, 1),
            "info":   info
        }

    except Exception as e:
        print(f"  ERRO GERAL dial {indice+1}: {e}")
        return {
            "digito": None,
            "angulo": None,
            "info":   f"dial {indice+1}: erro - {str(e)}"
        }


# ─────────────────────────────────────────────
# FUNÇÃO PRINCIPAL (compatibilidade com main.py)
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
