import tensorrt as trt

onnx_file_name = 'mnist.onnx'
tensorrt_file_name = 'mnist.trt'
fp16_mode = True
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

# Builder and network creation
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(EXPLICIT_BATCH)
parser = trt.OnnxParser(network, TRT_LOGGER)

# Create builder config
builder_config = builder.create_builder_config()

# Set FP16 mode
if fp16_mode:
    builder_config.set_flag(trt.BuilderFlag.FP16)

# Parse ONNX model
with open(onnx_file_name, 'rb') as model:
    if not parser.parse(model.read()):
        for error in range(parser.num_errors()):
            print(parser.get_error(error))

# Build CUDA engine
engine = builder.build_serialized_network(network, builder_config)


# Save the TensorRT engine to file
with open(tensorrt_file_name, 'wb') as f:
    f.write(engine)
