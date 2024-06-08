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
                             T.ToTensor(),
                             T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


class Output(BaseModel):
    
    output : int

@app.post('/prediction/')
def prediction(file : bytes = File(...)):

    image=  Image.open(io.BytesIO(file)).convert('RGB')
    image = input_transform(image).unsqueeze(0)
    output = model(image)
    output = torch.argmax(output,dim=1).item()

    return Output(output=output)

    
    
    
    