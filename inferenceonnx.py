import onnxruntime as ort
import numpy as np



providers = ["CUDAExecutionProvider"]

ort_session = ort.InferenceSession("alexnet.onnx",providers=providers)



outputs = ort_session.run(None, {"input":np.random.randn(10,3,224,224).astype(np.float32)})

print(outputs[0])
