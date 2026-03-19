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
                # lê colorido (BGR) — não converte para cinza aqui
                dial_bgr   = cv2.imdecode(arr, cv2.IMREAD_COLOR)

                if dial_bgr is None:
                    digitos.append(None)
                    angulos.append(None)
                    infos.append(f"dial {i+1}: imagem invalida")
                    continue

                resultado = ler_dial_individual(dial_bgr, orientacao, i)
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
                "erro":    "Nao foi possivel ler todos os dials."
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
