import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import onnx
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F

train_dataset = torchvision.datasets.MNIST(root='./fastapi_onnx_MNIST',
                      train=True,
                      download=True,
                      transform= T.ToTensor())
test_dataset =  torchvision.datasets.MNIST(root='./fastapi_onnx_MNIST',
                     train=False,
                     download=True,
                     transform=T.ToTensor())

train_loader = DataLoader(train_dataset,batch_size=32,shuffle=True)
device = torch.device('cuda:0')
learning_rate = 0.0001
epochs = 15
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.dropout = nn.Dropout2d(0.25)
        # (입력 뉴런, 출력 뉴런)
        self.fc1 = nn.Linear(3136, 1000)    # 7 * 7 * 64 = 3136
        self.fc2 = nn.Linear(1000, 10)
    
    def forward(self, input):
        x = self.conv1(input)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
    
    
model = CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr = learning_rate)
criterion = nn.CrossEntropyLoss()
model.train()

for epoch in range(epochs):
    
    for data,target in train_loader:

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
