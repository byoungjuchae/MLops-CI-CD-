import onnx
import onnxruntime as ort
import numpy 
from fastapi import FastAPI, File
from pydantic import BaseModel
import torch
import torchvision.models as models
from PIL import Image
import io
import torchvision.transforms as T
import numpy as np
providers = ["CUDAExecutionProvider"]

input_transform = T.Compose([T.Resize(224),
                             T.ToTensor()])



model = models.resnet18(pretrained=True).cuda()
dummy_input = torch.randn(1,3,224,224,device='cuda')
torch.onnx.export(model,dummy_input,'resnet.onnx',verbose=True,input_names=["input"],output_names=["output"])
app = FastAPI()



@app.post('/predict/')
def predict(file : bytes = File(...)):
    
    image=  Image.open(io.BytesIO(file)).convert('RGB')
    # image = np.array(image)

    image = input_transform(image).unsqueeze(0)
    # image = input_transform(image)
    model = ort.InferenceSession('resnet.onnx',providers=providers)

    output = model.run(None,{"input":image.numpy()})

        
    output = np.argmax(output[0],axis=1).item()
    
    return  output


