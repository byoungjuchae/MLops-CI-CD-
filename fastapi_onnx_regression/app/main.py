from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, FileResponse
from PIL import Image
import cv2
import io 





app = FastAPI()

@app.post("/predict/")
def predict(file: UploadFile = File(...)):
    file_bytes = file.file.read()
    image = Image.open(io.BytesIO(file_bytes))
    # new_image = prepare_image(image)
    # result = predict(image)
    #new_image.save(bytes_image, format='PNG')
    return FileResponse(file_bytes)