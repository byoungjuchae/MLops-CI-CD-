from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, FileResponse
from PIL import Image
import onnxruntime as ort
import cv2
import io
import torch 
import torchvision.transforms as T
import numpy as np

providers = ['CUDAExecutionProvider']

app = FastAPI()
transform = T.ToTensor()
ort_session = ort.InferenceSession('esrgan.onnx',providers=providers)
@app.post("/predict/")
def predict(file: UploadFile = File(...)):
    file_bytes = file.file.read()
    image = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)

    image = transform(image).unsqueeze(0)
    image = image.numpy()
    inputs = {ort_session.get_inputs()[0].name: image}

   
    output = ort_session.run(None,inputs)

    output_image =  output[0][0].transpose(1, 2, 0).astype(np.float32)*255.0

    _, img_encoded = cv2.imencode('.jpg', output_image)
   
    # new_image = prepare_image(image)
    # result = predict(image)
    #new_image.save(bytes_image, format='PNG')
    return StreamingResponse(io.BytesIO(img_encoded.tobytes()), media_type="image/jpeg")