from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import anthropic
import base64
import json
import os
from datetime import datetime

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

LEITURAS_FILE = "leituras.json"
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

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

@app.post("/ler-medidor")
async def ler_medidor(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        b64 = base64.standard_b64encode(contents).decode("utf-8")

        # detecta tipo da imagem
        media_type = file.content_type or "image/jpeg"

        prompt = """Você está vendo a foto de um medidor de energia elétrica analógico com 4 dials (mostrador de ponteiro).

Leia cada dial da esquerda para a direita seguindo esta regra:
- Veja entre quais dois números o ponteiro está apontando
- Devolva SEMPRE o menor dos dois números
- Só devolva o maior se o ponteiro estiver EXATAMENTE em cima dele

Devolva APENAS um JSON neste formato, sem texto adicional:
{"digitos": [D1, D2, D3, D4], "leitura": DDDD}

Onde D1 é o primeiro dial (esquerda), D2 o segundo, D3 o terceiro, D4 o quarto (direita).
DDDD é a leitura completa como número inteiro."""

        message = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=256,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ],
                }
            ],
        )

        resposta = message.content[0].text.strip()
        print(f"Resposta IA: {resposta}")

        # extrai o JSON da resposta
        inicio = resposta.find("{")
        fim    = resposta.rfind("}") + 1
        dados  = json.loads(resposta[inicio:fim])

        return {
            "digitos":     dados["digitos"],
            "leitura_kwh": dados["leitura"]
        }

    except Exception as e:
        print(f"ERRO: {e}")
        raise HTTPException(status_code=500, detail=f"Erro: {str(e)}")

@app.post("/salvar-leitura")
def salvar_leitura(req: LeituraRequest):
    try:
        leituras = carregar_leituras()
        nova = {
            "data": datetime.now().strftime("%Y-%m-%d"),
            "hora": datetime.now().strftime("%H:%M"),
            "leitura_kwh": req.leitura_kwh
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
