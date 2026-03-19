from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import json
import os
import re
from datetime import datetime


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
    chave = os.environ.get("ANTHROPIC_API_KEY", "")
    return {
        "status": "ok",
        "chave_configurada": bool(chave),
        "chave_prefixo": chave[:10] if chave else "VAZIA"
    }

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
        import anthropic

        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise HTTPException(
                status_code=500,
                detail="ANTHROPIC_API_KEY não configurada no servidor."
            )

        client = anthropic.Anthropic(api_key=api_key)

        contents = await file.read()
        b64 = base64.standard_b64encode(contents).decode("utf-8")
        media_type = file.content_type or "image/jpeg"

        message = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=256,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": b64
                        }
                    },
                    {
                        "type": "text",
                        "text": (
                            "Você está vendo a foto de um medidor de energia elétrica analógico com 4 dials. "
                            "Leia cada dial da esquerda para a direita. "
                            "Para cada dial: veja entre quais dois números o ponteiro está. "
                            "Retorne SEMPRE o menor dos dois. "
                            "Só retorne o maior se o ponteiro estiver exatamente em cima dele. "
                            "Responda APENAS com JSON neste formato, sem mais nada: "
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
