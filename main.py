from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from leitor import ler_medidor

app = FastAPI(title="Leitor de Medidor de Luz")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/ler-medidor")
async def endpoint(file: UploadFile = File(...)):
    img_bytes = await file.read()
    resultado = ler_medidor(img_bytes)
    if "erro" in resultado:
        return JSONResponse(resultado, status_code=422)
    return JSONResponse(resultado)