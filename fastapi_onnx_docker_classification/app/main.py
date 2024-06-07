import torch
import torchvision.models as models
import onnx
import onnxruntime as ort
import torchvision.transforms as T
from fastapi import FastAPI, File
import numpy as np
from PIL import Image
import io
app = FastAPI()
providers = ["CUDAExecutionProvider"]
input_transform = T.Compose([T.Resize(224),
                             T.ToTensor(),
                             T.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])

@app.post('/predict/')
async def predicton(file : bytes = File(...)) :
    
    
    image = Image.open(io.BytesIO(file))
    image = input_transform(image).unsqueeze(0)
    image = image.numpy()
    
    model = ort.InferenceSession('resnet.onnx',providers=providers)
    output = model.run(None,{"input":image.numpy()})
    output = np.argmax(output[0],axis=1).item()
    
    
    