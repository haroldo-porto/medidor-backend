from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import numpy as np
import base64
import json
import os
from datetime import datetime

from leitor import ler_dial_individual, recortar_dials, preprocessar

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LEITURAS_FILE = "leituras.json"


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
# HEALTH CHECK
# ─────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "message": "Medidor backend rodando"}


# ─────────────────────────────────────────────
# ENDPOINT 1: PROCESSAR FOTO
# Divide em 4 dials, aplica cinza suave, retorna base64
# ─────────────────────────────────────────────
@app.post("/processar-dials")
async def processar_dials(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(
                status_code=400,
                detail="Imagem inválida ou corrompida."
            )

        # Redimensiona se muito grande
        h, w = img.shape[:2]
        if w > 1400:
            scale = 1400 / w
            img = cv2.resize(img, (1400, int(h * scale)))

        # Pré-processa para detecção de círculos
        gray, _ = preprocessar(img)

        # Recorta os 4 dials
        dials, centros = recortar_dials(img, gray)

        if len(dials) < 4:
            raise HTTPException(
                status_code=400,
                detail="Não encontrei 4 dials. Tente foto mais próxima e centralizada."
            )

        dials_base64 = []
        for dial_img in dials:
            try:
                # ── PIPELINE DE COR SUAVE (como na foto de referência) ──

                # 1) Converte para cinza
                dial_gray = cv2.cvtColor(dial_img, cv2.COLOR_BGR2GRAY)

                # 2) CLAHE suave para realçar contraste local sem estourar
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                dial_gray = clahe.apply(dial_gray)

                # 3) Ajuste de brilho/contraste global
                #    alpha=1.2 = leve aumento de contraste
                #    beta=-10  = levemente mais escuro (fundo fica cinza escuro)
                dial_adj = cv2.convertScaleAbs(dial_gray, alpha=1.2, beta=-10)

                # 4) Suaviza ruído levemente (mantém bordas dos números)
                dial_adj = cv2.GaussianBlur(dial_adj, (3, 3), 0)

                # Codifica em PNG em tons de cinza suave
                _, buf = cv2.imencode('.png', dial_adj)
                dials_base64.append(base64.b64encode(buf).decode('utf-8'))

            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Erro ao processar dial: {str(e)}"
                )

        return {"dials": dials_base64, "total": len(dials_base64)}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")


# ─────────────────────────────────────────────
# ENDPOINT 2: LER DÍGITOS
# Recebe os 4 dials em base64, lê cada um
# ─────────────────────────────────────────────
@app.post("/ler-dials")
async def ler_dials(
    dial0: str = Form(...),
    dial1: str = Form(...),
    dial2: str = Form(...),
    dial3: str = Form(...),
):
    try:
        dials_b64   = [dial0, dial1, dial2, dial3]
        orientacoes = ["anticlockwise", "clockwise", "anticlockwise", "clockwise"]
        digitos     = []
        angulos     = []
        infos       = []

        for i, (b64, orientacao) in enumerate(zip(dials_b64, orientacoes)):
            try:
                dial_bytes = base64.b64decode(b64)
                arr        = np.frombuffer(dial_bytes, np.uint8)
                dial_img   = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)

                if dial_img is None:
                    digitos.append(None)
                    angulos.append(None)
                    infos.append(f"dial {i+1}: imagem inválida")
                    continue

                resultado = ler_dial_individual(dial_img, orientacao, i)
                digitos.append(resultado.get("digito"))
                angulos.append(resultado.get("angulo"))
                infos.append(resultado.get("info", f"dial {i+1}: ok"))

            except Exception as e:
                digitos.append(None)
                angulos.append(None)
                infos.append(f"dial {i+1}: erro - {str(e)}")

        if any(d is None for d in digitos):
            return {
                "digitos": digitos,
                "angulos": angulos,
                "infos":   infos,
                "erro":    "Não foi possível ler todos os dials."
            }

        leitura = int("".join(str(d) for d in digitos))
        return {
            "digitos":     digitos,
            "angulos":     angulos,
            "leitura_kwh": leitura,
            "infos":       infos
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")


# ─────────────────────────────────────────────
# ENDPOINT 3: SALVAR LEITURA
# ─────────────────────────────────────────────
@app.post("/salvar-leitura")
async def salvar_leitura(req: LeituraRequest):
    try:
        leituras = carregar_leituras()
        nova = {
            "data":        datetime.now().strftime("%Y-%m-%d"),
            "hora":        datetime.now().strftime("%H:%M"),
            "leitura_kwh": req.leitura_kwh
        }
        leituras.append(nova)
        salvar_leituras_arquivo(leituras)
        return {"sucesso": True, "leitura": nova}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao salvar: {str(e)}")


# ─────────────────────────────────────────────
# ENDPOINT 4: HISTÓRICO
# ─────────────────────────────────────────────
@app.get("/historico")
def historico():
    try:
        return {"leituras": carregar_leituras()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro: {str(e)}")
