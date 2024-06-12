MLops CI/CD 

This repository involves MLops's CI/CD stage.
I use docker,fastapi, onnx, jenkins to make CI/CD pipeline. 

-----------------------------------------------------------
What involves in the each Folder?

- fastapi : to use fastapi tutorial  --> Also, it just use dummy model
- fastapi_onnx_classifcation: to use fastapi and onnx tutorial (level 1) --> Also, it just use dummy model
- fastapi_onnx_MNIST : Upgrade to use fastapi and onnx tutorial (level 2) --> It is possible to train and evaluate model
- fastapi_onnx_docker_jenkins: To make CI/CD pipeline simply. But I don't upload the docker image to hub. 

-----------------------------------------------------------
To do

1. fastapi_onnx_docker_jenkins folder converts the model into the MNIST pretrained model.
2. fastapi_onnx_regression model such as super resolution.
3. Using diffusion model 
4. Using TensoRT

Until 2024/06/21