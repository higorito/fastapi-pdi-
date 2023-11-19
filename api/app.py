'''

    Projeto de PDI - FastAPI + OpenCV 

    Aluno: Higor Pereira Silva


    Para instalar as dependências do projeto, execute o comando abaixo no terminal:
    pip install -r requirements.txt


    Para executar o projeto, execute o comando abaixo no terminal:
    uvicorn main:app --reload
    Ou execute o arquivo main.py no seu editor de código.

    Para acessar a documentação da API roda e digite <endereço da API>/docs
'''


import cv2
import numpy as np
import base64
from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse


app = FastAPI()

async def process_image(contents):
    nparr = np.frombuffer(contents, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

async def process_grayscale_image(contents):
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    return image

async def encode_image(image):
    if isinstance(image, np.ndarray):
        _, img_encoded = cv2.imencode('.png', image)
        img_base64 = base64.b64encode(img_encoded.tobytes())
        return img_base64.decode()
    else:
        print("Error: Não é um array numpy")
        return None

#-------------------BINARIZAÇÃO-------------------#

async def aplicar_otsu_binarization(image):
    _, binarized_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binarized_image


#-------------------SEGMENTAÇÃO-POR-BORDA-------------------#
async def aplicar_canny_filtro(image):
    return cv2.Canny(image, 50, 150)

async def aplicar_roberts_filtro(image):
    altura, largura = image.shape[:2]
    nova_altura = altura // 2
    nova_largura = largura // 2
    imagem_redimensionada = cv2.resize(image, (nova_largura, nova_altura))

    roberts_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
    roberts_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
    
    roberts_x_result = cv2.filter2D(imagem_redimensionada, -1, roberts_x)
    roberts_y_result = cv2.filter2D(imagem_redimensionada, -1, roberts_y)

    return roberts_x_result, roberts_y_result

# async def aplicar_sobel_filtro(image):
#     altura, largura = image.shape[:2]
#     nova_altura = altura // 2
#     nova_largura = largura // 2
#     imagem_redimensionada = cv2.resize(image, (nova_largura, nova_altura))

#     sobel_x = cv2.Sobel(imagem_redimensionada, cv2.CV_64F, 1, 0, ksize=3)
#     sobel_x = np.absolute(sobel_x)
#     sobel_x = np.uint8(sobel_x)

#     sobel_y = cv2.Sobel(imagem_redimensionada, cv2.CV_64F, 0, 1, ksize=3)
#     sobel_y = np.absolute(sobel_y)
#     sobel_y = np.uint8(sobel_y)

#     return imagem_redimensionada, sobel_x, sobel_y

async def aplicar_sobel_filtro(image):
    altura, largura = image.shape[:2]
    nova_altura = altura // 2
    nova_largura = largura // 2
    imagem_redimensionada = cv2.resize(image, (nova_largura, nova_altura))

    sobel_x = cv2.Sobel(imagem_redimensionada, cv2.CV_64F, 1, 0, ksize=3)
    sobel_x = np.absolute(sobel_x)
    sobel_x = np.uint8(sobel_x)

    sobel_y = cv2.Sobel(imagem_redimensionada, cv2.CV_64F, 0, 1, ksize=3)
    sobel_y = np.absolute(sobel_y)
    sobel_y = np.uint8(sobel_y)

    return sobel_x, sobel_y

# async def aplicar_prewitt_filtro(image):
#     altura, largura = image.shape[:2]
#     nova_altura = altura // 2
#     nova_largura = largura // 2
#     imagem_redimensionada = cv2.resize(image, (nova_largura, nova_altura))

#     prewitt_x = cv2.filter2D(imagem_redimensionada, -1, np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]))
#     prewitt_y = cv2.filter2D(imagem_redimensionada, -1, np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]))

#     return imagem_redimensionada, prewitt_x, prewitt_y

async def aplicar_prewitt_filtro(image):
    altura, largura = image.shape[:2]
    nova_altura = altura // 2
    nova_largura = largura // 2
    imagem_redimensionada = cv2.resize(image, (nova_largura, nova_altura))

    prewitt_x = cv2.filter2D(imagem_redimensionada, -1, np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]))
    prewitt_y = cv2.filter2D(imagem_redimensionada, -1, np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]))

    return prewitt_x, prewitt_y

#-------------------SEGMENTAÇÃO-POR-REGIÃO-------------------#
async def aplicar_watershed(image):
    imagem_cinza = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imagem_desfocada = cv2.GaussianBlur(imagem_cinza, (5, 5), 0)
    _, imagem_binaria = cv2.threshold(imagem_desfocada, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    contornos, _ = cv2.findContours(imagem_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    marcadores = np.zeros(imagem_cinza.shape, dtype=np.int32)
    for i in range(len(contornos)):
        cv2.drawContours(marcadores, contornos, i, i+1, -1)
    cv2.circle(marcadores, (5,5), 3, (255,255,255), -1)

    for i, contorno in enumerate(contornos):
        cv2.drawContours(marcadores, [contorno], 0, i+1, -1)

    cv2.watershed(image, marcadores)

    for cor in range(1, len(contornos) + 1):
        image[marcadores == cor] = [0, 0, 255]  

    return image


#-----------------------------------------ROTAS-----------------------------------------------#
@app.get("/")
async def root():
    return {"message": "Esta é a API de processamento de imagens"}


@app.post("/uploadfile/")
async def upload_file(file: UploadFile):
    contents = await file.read()
    image = await process_grayscale_image(contents)
    img_base64 = await encode_image(image)
    return JSONResponse(content={"image": img_base64}, media_type="application/json")

@app.post("/aplicar-filtro/otsu/")
async def aplicar_otsu_filtro(file: UploadFile):
    contents = await file.read()
    image = await process_grayscale_image(contents)

    if isinstance(image, np.ndarray):
        otsu_binarized_image = await aplicar_otsu_binarization(image)
        otsu_binarized_base64 = await encode_image(otsu_binarized_image)

        if otsu_binarized_base64 is not None:
            return JSONResponse(content={"otsu_segmentation": otsu_binarized_base64}, media_type="application/json")
                                                #sei q isso é gambiarra, mas não consegui fazer de outra forma rs
    
    return JSONResponse(content={"error": "O processamento da imagem falhou"}, status_code=500)



@app.post("/aplicar-filtro/canny/")
async def aplicar_canny_filtro_route(file: UploadFile):
    contents = await file.read()
    image = await process_grayscale_image(contents)
    canny_segmentation = await aplicar_canny_filtro(image)
    canny_base64 = await encode_image(canny_segmentation)
    return JSONResponse(content={"canny_segmentation": canny_base64}, media_type="application/json")


@app.post("/aplicar-filtro/watershed/")
async def aplicar_watershed_route(file: UploadFile):
    contents = await file.read()
    image = await process_image(contents)
    segmented_image = await aplicar_watershed(image)
    segmented_base64 = await encode_image(segmented_image)
    return JSONResponse(content={"watershed_segmentation": segmented_base64}, media_type="application/json")


# @app.post("/aplicar-filtro/roberts/")
# async def aplicar_roberts_filtro_route(file: UploadFile):
#     contents = await file.read()
#     image = await process_image(contents)
#     original, roberts_x_result, roberts_y_result = await aplicar_roberts_filtro(image)

#     original_base64 = await encode_image(original)
#     roberts_x_base64 = await encode_image(roberts_x_result)
#     roberts_y_base64 = await encode_image(roberts_y_result)

#     return JSONResponse(content={
#         "original_image": original_base64,
#         "roberts_x_segmentation": roberts_x_base64,
#         "roberts_y_segmentation": roberts_y_base64
#     }, media_type="application/json")

@app.post("/aplicar-filtro/roberts_x/")
async def aplicar_roberts_x_filtro_route(file: UploadFile):
    contents = await file.read()
    image = await process_image(contents)
    roberts_x_result, _ = await aplicar_roberts_filtro(image)

    roberts_x_base64 = await encode_image(roberts_x_result)

    return JSONResponse(content={
        "roberts_x_segmentation": roberts_x_base64
    }, media_type="application/json")

@app.post("/aplicar-filtro/roberts_y/")
async def aplicar_roberts_y_filtro_route(file: UploadFile):
    contents = await file.read()
    image = await process_image(contents)
    _, roberts_y_result = await aplicar_roberts_filtro(image)

    roberts_y_base64 = await encode_image(roberts_y_result)

    return JSONResponse(content={
        "roberts_y_segmentation": roberts_y_base64
    }, media_type="application/json")

# @app.post("/aplicar-filtro/sobel/")
# async def aplicar_sobel_filtro_route(file: UploadFile):
#     contents = await file.read()
#     image = await process_image(contents)
#     original, sobel_x_result, sobel_y_result = await aplicar_sobel_filtro(image)

#     original_base64 = await encode_image(original)
#     sobel_x_base64 = await encode_image(sobel_x_result)
#     sobel_y_base64 = await encode_image(sobel_y_result)

#     return JSONResponse(content={
#         "original_image": original_base64,
#         "sobel_x_segmentation": sobel_x_base64,
#         "sobel_y_segmentation": sobel_y_base64
#     }, media_type="application/json")

@app.post("/aplicar-filtro/sobel_x/")
async def aplicar_sobel_x_filtro_route(file: UploadFile):
    contents = await file.read()
    image = await process_image(contents)
    sobel_x_result, _ = await aplicar_sobel_filtro(image)

    sobel_x_base64 = await encode_image(sobel_x_result)

    return JSONResponse(content={
        "sobel_x_segmentation": sobel_x_base64
    }, media_type="application/json")

@app.post("/aplicar-filtro/sobel_y/")
async def aplicar_sobel_y_filtro_route(file: UploadFile):
    contents = await file.read()
    image = await process_image(contents)
    _, sobel_y_result = await aplicar_sobel_filtro(image)

    sobel_y_base64 = await encode_image(sobel_y_result)

    return JSONResponse(content={
        "sobel_y_segmentation": sobel_y_base64
    }, media_type="application/json")

# @app.post("/aplicar-filtro/prewitt/")
# async def aplicar_prewitt_filtro_route(file: UploadFile):
#     contents = await file.read()
#     image = await process_image(contents)
#     original, prewitt_x_result, prewitt_y_result = await aplicar_prewitt_filtro(image)

#     original_base64 = await encode_image(original)
#     prewitt_x_base64 = await encode_image(prewitt_x_result)
#     prewitt_y_base64 = await encode_image(prewitt_y_result)

#     return JSONResponse(content={
#         "original_image": original_base64,
#         "prewitt_x_segmentation": prewitt_x_base64,
#         "prewitt_y_segmentation": prewitt_y_base64
#     }, media_type="application/json")

@app.post("/aplicar-filtro/prewitt_x/")
async def aplicar_prewitt_x_filtro_route(file: UploadFile):
    contents = await file.read()
    image = await process_image(contents)
    prewitt_x_result, _ = await aplicar_prewitt_filtro(image)

    prewitt_x_base64 = await encode_image(prewitt_x_result)

    return JSONResponse(content={
        "prewitt_x_segmentation": prewitt_x_base64
    }, media_type="application/json")

@app.post("/aplicar-filtro/prewitt_y/")
async def aplicar_prewitt_y_filtro_route(file: UploadFile):
    contents = await file.read()
    image = await process_image(contents)
    _, prewitt_y_result = await aplicar_prewitt_filtro(image)

    prewitt_y_base64 = await encode_image(prewitt_y_result)

    return JSONResponse(content={
        "prewitt_y_segmentation": prewitt_y_base64
    }, media_type="application/json")