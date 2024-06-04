import torch
import torchvision


dummy_input = torch.randn(10,3,224,224,device='cuda')
model = torchvision.models.alexnet(pretrained=True).cuda()
dummy_output = model(dummy_input)


torch.onnx.export(model, dummy_input, "alexnet.onnx",verbose=True,input_names=["input"],output_names=["output"])