from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import os
from datetime import datetime
import base64
import httpx  # vamos chamar a IA de fora

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

@app.get("/")
def root():
    return {"status": "ok"}

# === AQUI É O ENDPOINT QUE LÊ A FOTO INTEIRA ===
@app.post("/ler-medidor")
async def ler_medidor(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        b64 = base64.b64encode(contents).decode("utf-8")

        # URL da IA de visão do Inner AI (exemplo genérico).
        # Você vai configurar essa URL no painel do Inner se/quando tiver.
        IA_URL = os.environ.get("IA_VISUAL_URL", "")

        if not IA_URL:
            raise HTTPException(
                status_code=500,
                detail="IA_VISUAL_URL não configurada no servidor."
            )

        payload = {
            "image_base64": b64,
            "prompt": (
                "Você está vendo a foto de um medidor de energia analógico "
                "com 4 dials. Leia os 4 dials da esquerda para a direita. "
                "Para cada dial: se o ponteiro estiver entre dois números, "
                "retorne SEMPRE o menor; só retorne o maior se o ponteiro "
                "estiver exatamente em cima dele. "
                "Responda apenas um JSON assim: "
                '{"digitos":[D1,D2,D3,D4],"leitura_kwh":DDDD}'
            )
        }

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(IA_URL, json=payload)
        if resp.status_code != 200:
            raise HTTPException(
                status_code=500,
                detail=f"Erro da IA ({resp.status_code}): {resp.text}"
            )

        dados = resp.json()

        # validação mínima
        if "digitos" not in dados or "leitura_kwh" not in dados:
            raise HTTPException(status_code=500, detail="Resposta inválida da IA.")

        return dados

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao ler medidor: {str(e)}")


@app.post("/salvar")
def salvar_leitura(req: LeituraRequest):
    try:
        leituras = carregar_leituras()
        agora     = datetime.now()
        nova = {
            "data":        agora.strftime("%d/%m/%Y"),
            "hora":        agora.strftime("%H:%M"),
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
