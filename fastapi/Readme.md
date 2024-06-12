This folder is to practice the fastapi tutorial.

It just use simple pretrained resnet model and make API to use fastapi.


Install environment

```bash
pip install -r requirements.txt
```

Start fastAPI and evaluate model 

```bash
uvicorn main:app --reload 
```

Go to 

'''
ip address/docs 
'''

<img src ='./docs/example.png'>
If you want to use docker 

```bash
docker build -t practice .
```

docker exec
