import torch
import torchvision.models as models
import onnx
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import onnxruntime as ort
import torchvision.transforms as T
from fastapi import FastAPI, File
import numpy as np
from PIL import Image
import io
from model import CNN

train_dataset = torchvision.datasets.MNIST(root= './fastapi_onnx_docker_jenkins',
                                           train=True,
                                           download=True,
                                           transform= T.ToTensor())
test_dataset = torchvision.datasets.MNIST(root='./fastapi_onnx_docker_jenkins',
                                          train=False,
                                          download=True,
                                          transform =T.ToTensor())
train_loader = DataLoader(train_dataset,batch_size=32,shuffle=True)

learing_rate =0.0001

device = torch.device('cuda:0')
models = CNN().to(device)
optimizer = optim.Adam(models.parameters(),lr=learning_Rate)
loss =  nn.CrossEntropyLoss()

app = FastAPI()
providers = ["CUDAExecutionProvider"]
input_transform = T.Compose([T.Resize(224),
                             T.ToTensor(),
                             T.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])


for epoch in range(epochs):
    
    for k,v in train_loader:
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    if epoch % 1 == 0:
        print("Train Step : {}\tLoss : {:3f}".format(epoch, loss.item()))
    
        
torch.onnx.export(model,data[0].unsqueeze(0),'mnist.onnx',export_params=True,verbose=True,input_names=["input"],output_names=["output"])


    
    