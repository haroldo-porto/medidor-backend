import anthropic
import base64
import cv2
import numpy as np

ORIENTACOES = ["anticlockwise", "clockwise", "anticlockwise", "clockwise"]

client = anthropic.Anthropic(api_key="SUA_CHAVE_ANTHROPIC")

def preprocessar(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray, gray
    except Exception:
        h, w = img.shape[:2]
        z = np.zeros((h, w), dtype=np.uint8)
        return z, z

def recortar_dials(img, gray):
    try:
        h, w = img.shape[:2]
        dw = w // 4
        dials = []
        centros = []
        for i in range(4):
            x1 = i * dw
            x2 = (i + 1) * dw if i < 3 else w
            crop = img[0:h, x1:x2]
            crop = cv2.resize(crop, (300, 300))
            dials.append(crop)
            centros.append(((x1 + x2) // 2, h // 2, dw // 2))
        return dials, centros
    except Exception:
        return [], []

def ler_dial_individual(dial_bgr, orientacao, indice):
    try:
        _, buf = cv2.imencode('.jpg', dial_bgr)
        b64 = base64.b64encode(buf.tobytes()).decode('utf-8')

        sentido = "anti-horário (números crescem para a esquerda)" if orientacao == "anticlockwise" else "horário (números crescem para a direita)"

        prompt = f"""Esta imagem mostra um dial de medidor de energia elétrica analógico.
O sentido de leitura deste dial é: {sentido}.
Regra: observe onde o ponteiro está apontando.
- Se estiver ENTRE dois números, devolva SEMPRE o número menor (o que ficou para trás).
- Só devolva o número maior se o ponteiro estiver EXATAMENTE em cima dele.
Responda APENAS com um único dígito (0 a 9), sem explicação."""

        response = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=10,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": b64
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }]
        )

        resposta = response.content[0].text.strip()
        print(f"[dial {indice+1}] orientacao={orientacao} -> resposta IA = '{resposta}'")

        digito = int(resposta[0]) if resposta and resposta[0].isdigit() else None

        return {
            "digito": digito,
            "angulo": None,
            "info": f"dial {indice+1}: IA leu '{resposta}' -> digito={digito}"
        }

    except Exception as e:
        print(f"ERRO dial {indice+1}: {e}")
        return {"digito": None, "angulo": None, "info": f"dial {indice+1}: erro - {str(e)}"}

def ler_medidor(img_bytes):
    try:
        arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return {"erro": "Imagem invalida."}

        h, w = img.shape[:2]
        if w > 1400:
            scale = 1400 / w
            img = cv2.resize(img, (1400, int(h * scale)))

        gray, _ = preprocessar(img)
        dials, centros = recortar_dials(img, gray)

        if len(dials) < 4:
            return {"erro": "Nao encontrei 4 dials."}

        digitos, angulos, infos = [], [], []

        for i, (dial_bgr, orientacao) in enumerate(zip(dials, ORIENTACOES)):
            resultado = ler_dial_individual(dial_bgr, orientacao, i)
            digitos.append(resultado.get("digito"))
            angulos.append(resultado.get("angulo"))
            infos.append(resultado.get("info"))

        if any(d is None for d in digitos):
            return {"erro": "Nao detectei todos os ponteiros.", "digitos": digitos, "angulos": angulos, "infos": infos}

        leitura = int("".join(str(d) for d in digitos))
        return {"digitos": digitos, "angulos": angulos, "leitura_kwh": leitura, "infos": infos}

    except Exception as e:
        return {"erro": f"Erro interno: {str(e)}"}
