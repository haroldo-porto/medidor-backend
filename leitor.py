import cv2
import numpy as np
import math

ORIENTACOES = ["anticlockwise", "clockwise", "anticlockwise", "clockwise"]

# Ângulos de referência: 0=topo, sentido horário
# Cada dígito ocupa 36 graus (360/10)
# 0 está no topo (0°), 1 está a 36°, 2 a 72°, etc.
ANG_REF = [i * 36.0 for i in range(10)]


def preprocessar(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
        gray  = clahe.apply(gray)
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
            raise ValueError("Nenhum circulo")

        circles = np.round(circles[0]).astype(int)

        ys    = [c[1] for c in circles]
        y_med = float(np.median(ys))
        y_tol = int(h * 0.20)
        circles = [c for c in circles if abs(c[1] - y_med) <= y_tol]

        if len(circles) < 4:
            raise ValueError("Menos de 4 circulos")

        circles = sorted(circles, key=lambda c: c[0])
        raios   = [c[2] for c in circles]
        r_med   = float(np.median(raios))
        if r_med <= 0:
            r_med = min_r

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

        escolhidos = []
        for coluna in colunas:
            ys_c   = [c[1] for c in coluna]
            y_mc   = float(np.median(ys_c))
            melhor = min(coluna, key=lambda c: abs(c[1] - y_mc))
            escolhidos.append(melhor)

        escolhidos = sorted(escolhidos, key=lambda c: c[0])

        dials   = []
        centros = []

        for cx, cy, r in escolhidos:
            half = int(r * 1.6)
            x1   = max(cx - half, 0)
            x2   = min(cx + half, w)
            y1   = max(cy - half, 0)
            y2   = min(cy + half, h)

            crop = img[y1:y2, x1:x2]
            crop = cv2.resize(crop, (300, 300))
            dials.append(crop)
            centros.append((cx, cy, r))

        if len(dials) != 4:
            raise ValueError("Nao formou 4 recortes")

        return dials, centros

    except Exception:
        h, w    = img.shape[:2]
        dials   = []
        centros = []
        dw = w // 4
        for i in range(4):
            x1   = i * dw
            x2   = (i + 1) * dw if i < 3 else w
            crop = img[0:h, x1:x2]
            crop = cv2.resize(crop, (300, 300))
            dials.append(crop)
            centros.append(((x1 + x2) // 2, h // 2, min(dw, h) // 2))
        return dials, centros


def preparar_para_ponteiro(dial_bgr):
    gray = cv2.cvtColor(dial_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray  = clahe.apply(gray)
    gray  = cv2.convertScaleAbs(gray, alpha=1.3, beta=-10)
    return gray


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
            score       = far_d * comprimento / max(near_d + 1, 1)
            ang         = math.degrees(math.atan2(px - cx, cy - py)) % 360
            candidatos.append((score, ang))

        if not candidatos:
            return None
        candidatos.sort(reverse=True)
        return candidatos[0][1]
    except Exception:
        return None


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


def detectar_angulo_ponteiro(dial_gray):
    try:
        h, w   = dial_gray.shape[:2]
        cx, cy = w // 2, h // 2
        r      = min(w, h) // 2

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (cx, cy), int(r * 0.82), 255, -1)
        cv2.circle(mask, (cx, cy), int(r * 0.14),   0, -1)

        blur   = cv2.GaussianBlur(dial_gray, (5, 5), 0)
        binary = cv2.adaptiveThreshold(
            blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=21, C=8
        )
        binary = cv2.bitwise_and(binary, mask)

        ang = detectar_por_linhas(binary, cx, cy, r)
        if ang is not None:
            return ang
        return detectar_por_fatias(binary, cx, cy, r)
    except Exception:
        return None


def digito_por_angulo_simples(angulo_ponteiro, orientacao):
    """
    Logica simples — exatamente o que o olho humano faz:
    1. Ajusta o sentido (anti-horario espelha o angulo)
    2. Ve em qual intervalo [N, N+1] o ponteiro cai
    3. Devolve SEMPRE o numero anterior N (o menor)
    4. So arredonda para N+1 se o ponteiro estiver colado (menos de 3 graus)
    """
    ang = angulo_ponteiro % 360.0

    # se anti-horario, espelha
    if orientacao == "anticlockwise":
        ang = (360.0 - ang) % 360.0

    for d in range(10):
        a1 = ANG_REF[d]
        a2 = ANG_REF[(d + 1) % 10]

        if a1 <= a2:
            no_intervalo = (a1 <= ang < a2)
        else:
            # intervalo que cruza 360->0 (entre 9 e 0)
            no_intervalo = (ang >= a1 or ang < a2)

        if no_intervalo:
            # distancia ate o proximo numero
            dist_prox = (a2 - ang) % 360.0
            # se estiver colado no proximo, arredonda para frente
            if dist_prox < 3.0:
                print(f"  colado no proximo {(d+1)%10} (falta {dist_prox:.1f} graus)")
                return (d + 1) % 10
            # regra principal: devolve o menor (o de tras)
            print(f"  entre {d} e {(d+1)%10}, falta {dist_prox:.1f} graus -> digito={d}")
            return d

    # nao deveria chegar aqui
    return 0


def ler_dial_individual(dial_bgr, orientacao, indice):
    try:
        print(f"\n{'='*50}")
        print(f"[dial {indice+1}] orientacao={orientacao}")

        dial_gray = preparar_para_ponteiro(dial_bgr)
        angulo    = detectar_angulo_ponteiro(dial_gray)
        print(f"  angulo ponteiro = {angulo}")

        if angulo is None:
            print("  ERRO: ponteiro nao detectado")
            return {
                "digito": None,
                "angulo": None,
                "info":   f"dial {indice+1}: ponteiro nao detectado"
            }

        # LOGICA SIMPLES: so angulo + sentido, sem Vision
        digito = digito_por_angulo_simples(angulo, orientacao)

        print(f"  >>> DIGITO FINAL = {digito}")

        return {
            "digito": digito,
            "angulo": round(angulo, 1),
            "info":   f"dial {indice+1}: ang={round(angulo,1)} -> {digito}"
        }

    except Exception as e:
        print(f"  ERRO GERAL dial {indice+1}: {e}")
        return {
            "digito": None,
            "angulo": None,
            "info":   f"dial {indice+1}: erro - {str(e)}"
        }


def ler_medidor(img_bytes):
    try:
        arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return {"erro": "Imagem invalida."}

        h, w = img.shape[:2]
        if w > 1400:
            scale = 1400 / w
            img   = cv2.resize(img, (1400, int(h * scale)))

        gray, _ = preprocessar(img)
        dials, centros = recortar_dials(img, gray)

        if len(dials) < 4:
            return {"erro": "Nao encontrei 4 dials."}

        digitos = []
        angulos = []
        infos   = []

        for i, (dial_bgr, orientacao) in enumerate(zip(dials, ORIENTACOES)):
            resultado = ler_dial_individual(dial_bgr, orientacao, i)
            digitos.append(resultado.get("digito"))
            angulos.append(resultado.get("angulo"))
            infos.append(resultado.get("info"))

        if any(d is None for d in digitos):
            return {
                "erro":    "Nao detectei todos os ponteiros.",
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
