This folder is to practice the fastapi tutorial and onnx export.

It just use simple pretrained resnet model and make API to use fastapi.

It is difference between fastapi folder and this folder. 
This folder involves the exporting the model as a onnx tool.
 
Install environment
'''bash
pip install -r requirements.txt
'''
Start fastAPI and evaluate model 
'''bash
uvicorn main:app --reload 
'''
Go to 
'''
ip address/docs 
'''
./docs/example.png
If you want to use docker 

```bash
docker build -t practice .
docker run -d -p 8000:8000 practice
```
