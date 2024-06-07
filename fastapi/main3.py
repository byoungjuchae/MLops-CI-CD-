from fastapi import FastAPI, File
import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from pydantic import BaseModel
import io


app = FastAPI()

model = models.resnet18(pretrained=True)

input_transform = T.Compose([T.Resize(224),
                             T.ToTensor()])


class Output(BaseModel):
    
    output : int

@app.post('/prediction/')
def prediction(file : bytes = File(...)):

    image=  Image.open(io.BytesIO(file)).convert('RGB')
    image = input_transform(image).unsqueeze(0)
    output = model(image)
    output = torch.argmax(output,dim=1).item()

    return Output(output=output)

    
    
    
    