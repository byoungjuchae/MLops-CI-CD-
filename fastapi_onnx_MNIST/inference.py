import onnxruntime as ort
from fastapi import FastAPI,File
import torchvision
import io
import torch
from PIL import Image
import numpy as np
import torchvision.transforms as T
providers = ["CUDAExectuionProvider"]

app = FastAPI()
model = ort.InferenceSession('mnist.onnx',providers=providers)
transform = T.Compose([T.Grayscale(),T.Resize(28),T.ToTensor()])
@app.post('/prediction/')
async def prediction(file : bytes = File(...)):
    
    image = Image.open(io.BytesIO(file))
    image = transform(image)
    image = image.unsqueeze(0)
    image=  image.numpy()
    with torch.no_grad():
        output = model.run(None,{"input":image})
    
    output = np.argmax(output[0],axis=1).item()
    
    return output
    