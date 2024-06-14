from model import RRDBNet
import torch
import onnx
import torch.nn as nn



def dynamic_shape(onnx_path):
    
    model = onnx.load(onnx_path)
    
    input_tensor = model.graph.input[0]
    
    input_tensor.type.tensor_type.shape.dim[2].dim_param = 'dynamic_height'
    input_tensor.type.tensor_type.shape.dim[3].dim_param = 'dynamic_width'
    
    onnx.checker.check_model(model)
    
    onnx.save(model, onnx_path)
models = RRDBNet(3, 3, 64, 23, gc=32).cuda()

models.load_state_dict(torch.load('RRDB_ESRGAN_x4.pth'))

data = torch.zeros((1,3,224,224)).cuda()
torch.onnx.export(models,data[0].unsqueeze(0),'esrgan.onnx',export_params=True,verbose=True,input_names=["input"],output_names=["output"])
dynamic_shape('esrgan.onnx')
