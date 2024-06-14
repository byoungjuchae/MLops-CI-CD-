import onnxruntime as ort
from fastapi import FastAPI, File
from PIL import Image
import io
from model import CNN
import torchvision.transforms as T

providers= ["CUDAExecutionProvider"]
input_transform = T.Compose([T.Resize(224),
                             T.ToTensor(),
                             T.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])
app = FastAPI()
model = CNN().cuda()

@app.post('/predict/')
async def predicton(file : bytes = File(...)) :
    
    
    image = Image.open(io.BytesIO(file))
    image = input_transform(image).unsqueeze(0)
    image = image.numpy()
    
    
    model = ort.InferenceSession('mnist.onnx',providers=providers)
    output = model.run(None,{"input":image.numpy()})
    output = np.argmax(output[0],axis=1).item()
    
    return {'output':output}
    