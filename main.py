from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import base64
import io
from leitor import ler_medidor, recortar_dials, preprocessar

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# ENDPOINT 1: PROCESSAR FOTO (dividir em 4 dials)
# ─────────────────────────────────────────────
@app.post("/processar-dials")
async def processar_dials(file: UploadFile = File(...)):
    """
    Recebe a foto inteira do medidor.
    Retorna os 4 dials processados em P&B como base64.
    """
    try:
        # Lê o arquivo
        contents = await file.read()
        arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Imagem inválida")

        # Redimensiona se muito grande
        h, w = img.shape[:2]
        if w > 1400:
            scale = 1400 / w
            img = cv2.resize(img, (1400, int(h * scale)))

        # Pré-processa
        gray, binary = preprocessar(img)

        # Recorta os 4 dials
        dials, centros = recortar_dials(img, gray)

        if len(dials) < 4:
            raise HTTPException(
                status_code=400,
                detail="Não encontrei 4 dials. Tente foto mais próxima."
            )

        # Converte cada dial para P&B e depois para base64
        dials_base64 = []
        for dial_img in dials:
            # Pré-processa o dial individual
            dial_gray = cv2.cvtColor(dial_img, cv2.COLOR_BGR2GRAY)
            dial_gray = cv2.bilateralFilter(dial_gray, 9, 75, 75)
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
            dial_gray = clahe.apply(dial_gray)
            dial_binary = cv2.adaptiveThreshold(
                dial_gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                blockSize=21, C=8
            )

            # Converte para PNG e depois base64
            _, buf = cv2.imencode('.png', dial_binary)
            dial_base64 = base64.b64encode(buf).decode('utf-8')
            dials_base64.append(dial_base64)

        return {"dials": dials_base64}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar: {str(e)}")


# ─────────────────────────────────────────────
# ENDPOINT 2: LER DÍGITOS (cada dial separadamente)
# ─────────────────────────────────────────────
@app.post("/ler-dials")
async def ler_dials(
    dial0: UploadFile = File(...),
    dial1: UploadFile = File(...),
    dial2: UploadFile = File(...),
    dial3: UploadFile = File(...),
):
    """
    Recebe os 4 dials já processados em P&B.
    Retorna os 4 dígitos lidos.
    """
    try:
        dials_files = [dial0, dial1, dial2, dial3]
        dials_bytes = []

        # Lê cada dial
        for dial_file in dials_files:
            contents = await dial_file.read()
            dials_bytes.append(contents)

        # Monta uma imagem "fake" com os 4 dials lado a lado
        # para passar para a função ler_medidor
        dials_imgs = []
        for dial_bytes in dials_bytes:
            arr = np.frombuffer(dial_bytes, np.uint8)
            dial_img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
            if dial_img is None:
                raise HTTPException(status_code=400, detail="Dial inválido")
            dials_imgs.append(dial_img)

        # Cria uma imagem "fake" concatenando os 4 dials
        # (só para passar pela função ler_medidor)
        h = dials_imgs[0].shape[0]
        w = dials_imgs[0].shape[1]
        fake_img = np.hstack(dials_imgs)

        # Converte de volta para BGR para passar para ler_medidor

