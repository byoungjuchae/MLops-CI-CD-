import onnxruntime as ort
from fastapi import FastAPI, File
from PIL import Image
import io



app = FastAPI()


@app.post('/predict/')
async def predicton(file : bytes = File(...)) :
    
    
    image = Image.open(io.BytesIO(file))
    image = input_transform(image).unsqueeze(0)
    image = image.numpy()
    
    
    model = ort.InferenceSession('mnist.onnx',providers=providers)
    output = model.run(None,{"input":image.numpy()})
    output = np.argmax(output[0],axis=1).item()
    
    return {'output':output}
    