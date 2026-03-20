from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import anthropic
import base64
import json
import os
import tempfile
import re
from datetime import datetime
from PIL import Image, ImageEnhance
import numpy as np
from scipy import ndimage

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LEITURAS_FILE = "leituras.json"
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))


# ─────────────────────────────────────────────
# PRÉ-PROCESSAMENTO DA IMAGEM
# ─────────────────────────────────────────────

def pre_processar_imagem(image_bytes: bytes) -> str:
    """
    Recebe os bytes da imagem original e retorna base64 da imagem processada:
    - Escala de cinza com contraste forte nos números (pretos)
    - Ponteiros pintados de vermelho
    - Imagem ampliada 4x para melhor leitura
    """
    # Salvar bytes em arquivo temporário para abrir com PIL
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_in:
        tmp_in.write(image_bytes)
        tmp_in_path = tmp_in.name

    try:
        img = Image.open(tmp_in_path).convert("RGB")
        width, height = img.size

        # 1. Ampliar 4x para melhor resolução
        img_big = img.resize((width * 4, height * 4), Image.LANCZOS)
        W, H = img_big.size

        # 2. Escala de cinza com contraste e nitidez fortes
        gray = img_big.convert("L")
        gray = ImageEnhance.Contrast(gray).enhance(3.0)
        gray = ImageEnhance.Sharpness(gray).enhance(4.0)
        gray_arr = np.array(gray)

        # Escurecer pixels já escuros (números ficam bem pretos)
        arr = np.where(gray_arr < 160, gray_arr * 0.3, gray_arr)
        arr = np.clip(arr, 0, 255).astype(np.uint8)

        # 3. Base RGB para colorir ponteiros
        rgb_arr = np.array(img_big.convert("RGB"))

        # Converter para cinza processada no RGB também
        gray_rgb = np.stack([arr, arr, arr], axis=-1)
        rgb_arr = gray_rgb  # usa versão cinza como base

        # 4. Detectar e pintar ponteiros de vermelho dentro de cada dial
        # Centros (cx, cy) e raio de cada dial na imagem 4x
        # Ajustados para imagem original ~378x291px
        dials_coords = [
            (int(95  * (width / 378) * 4), int(130 * (height / 291) * 4), int(55 * (width / 378) * 4)),
            (int(178 * (width / 378) * 4), int(130 * (height / 291) * 4), int(55 * (width / 378) * 4)),
            (int(242 * (width / 378) * 4), int(130 * (height / 291) * 4), int(55 * (width / 378) * 4)),
            (int(320 * (width / 378) * 4), int(130 * (height / 291) * 4), int(55 * (width / 378) * 4)),
        ]

        for (cx, cy, r) in dials_coords:
            Y, X = np.ogrid[:H, :W]
            circle_mask = (X - cx)**2 + (Y - cy)**2 <= (r * 0.75)**2
            dark_mask = arr < 90
            ponteiro_mask = circle_mask & dark_mask
            ponteiro_mask = ndimage.binary_erosion(ponteiro_mask, iterations=2)
            ponteiro_mask = ndimage.binary_dilation(ponteiro_mask, iterations=4)
            rgb_arr[ponteiro_mask] = [220, 30, 30]

        # 5. Salvar resultado e converter para base64
        img_resultado = Image.fromarray(rgb_arr.astype(np.uint8))
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_out:
            img_resultado.save(tmp_out.name)
            tmp_out_path = tmp_out.name

        with open(tmp_out_path, "rb") as f:
            b64 = base64.standard_b64encode(f.read()).decode("utf-8")

        os.unlink(tmp_out_path)
        return b64

    finally:
        os.unlink(tmp_in_path)


# ─────────────────────────────────────────────
# FUNÇÕES AUXILIARES
# ─────────────────────────────────────────────

def carregar_leituras():
    if not os.path.exists(LEITURAS_FILE):
        return []
    try:
        with open(LEITURAS_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []


def salvar_leituras_arquivo(leituras):
    with open(LEITURAS_FILE, "w") as f:
        json.dump(leituras, f, ensure_ascii=False, indent=2)


class LeituraRequest(BaseModel):
    leitura_kwh: int


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/status")
def status():
    chave = os.environ.get("ANTHROPIC_API_KEY", "")
    return {
        "chave_configurada": bool(chave),
        "chave_prefixo": chave[:8] if chave else "VAZIA"
    }


@app.post("/ler-medidor")
async def ler_medidor(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        # Pré-processar imagem: cinza + contraste + ponteiros vermelhos
        b64_processada = pre_processar_imagem(contents)

        # Enviar imagem processada para Claude com instruções detalhadas
        message = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": b64_processada
                        }
                    },
                    {
                        "type": "text",
                        "text": (
                            "Você está vendo um medidor de energia elétrica analógico com 4 dials. "
                            "A imagem foi pré-processada: está em tons de cinza com contraste forte nos números "
                            "e os PONTEIROS estão pintados de VERMELHO para facilitar a leitura. "
                            "\n\n"
                            "SENTIDO DE LEITURA DE CADA DIAL:\n"
                            "- Dial 1 (mais à esquerda): sentido ANTI-HORÁRIO\n"
                            "- Dial 2: sentido HORÁRIO\n"
                            "- Dial 3: sentido ANTI-HORÁRIO\n"
                            "- Dial 4 (mais à direita): sentido HORÁRIO\n"
                            "\n"
                            "REGRA DE LEITURA (aplique para cada dial no seu respectivo sentido):\n"
                            "1. Observe onde a ponta do ponteiro VERMELHO está apontando.\n"
                            "2. Identifique entre quais dois números consecutivos o ponteiro está.\n"
                            "3. Retorne SEMPRE o MENOR dos dois números.\n"
                            "4. Só retorne o maior se o ponteiro estiver EXATAMENTE sobre ele.\n"
                            "\n"
                            "Responda APENAS com JSON neste formato, sem mais nada:\n"
                            '{"digitos":[D1,D2,D3,D4],"leitura_kwh":DDDD}'
                        )
                    }
                ]
            }]
        )

        texto = message.content[0].text.strip()
        match = re.search(r'\{.*\}', texto, re.DOTALL)
        if not match:
            raise HTTPException(status_code=500, detail=f"IA não retornou JSON: {texto}")

        dados = json.loads(match.group())

        if "digitos" not in dados or "leitura_kwh" not in dados:
            raise HTTPException(status_code=500, detail="JSON inválido da IA.")

        return dados

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro: {str(e)}")


@app.post("/salvar")
def salvar_leitura(req: LeituraRequest):
    try:
        leituras = carregar_leituras()
        agora = datetime.now()
        nova = {
            "data": agora.strftime("%d/%m/%Y"),
            "hora": agora.strftime("%H:%M"),
            "leitura_kwh": req.leitura_kwh,
        }
        leituras.append(nova)
        salvar_leituras_arquivo(leituras)
        return {"sucesso": True, "leitura": nova}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao salvar: {str(e)}")


@app.get("/historico")
def historico():
    try:
        return {"leituras": carregar_leituras()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro: {str(e)}")
