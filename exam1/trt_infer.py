import time
import numpy as np
import tensorrt as trt
from cuda import cudart
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

start = time.time()

width, height = 28, 28
engineString = 'model_trt.plan'

logger = trt.Logger(trt.Logger.WARNING)

with open(engineString, "rb") as f:
    engineString = f.read()

engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
nIO = engine.num_io_tensors
lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]
nInput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.INPUT)

context = engine.create_execution_context()
context.set_input_shape(lTensorName[0], [1, 1, height, width])
for i in range(nIO):
    print("[%2d]%s->" % (i, "Input " if i < nInput else "Output"), engine.get_tensor_dtype(lTensorName[i]),
          engine.get_tensor_shape(lTensorName[i]), context.get_tensor_shape(lTensorName[i]), lTensorName[i])

trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
test_dataset = DataLoader(datasets.MNIST("datasets/", train=False, download=False, transform=trans),
                          batch_size=1, shuffle=False)

correct = 0
for index, (data, label) in enumerate(test_dataset):
    bufferH = []
    bufferH.append(np.ascontiguousarray(data))
    for i in range(nInput, nIO):
        bufferH.append(
            np.empty(context.get_tensor_shape(lTensorName[i]),
                     dtype=trt.nptype(engine.get_tensor_dtype(lTensorName[i]))))
    bufferD = []
    for i in range(nIO):
        bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

    for i in range(nInput):
        cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes,
                          cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

    for i in range(nIO):
        context.set_tensor_address(lTensorName[i], int(bufferD[i]))

    context.execute_async_v3(0)

    for i in range(nInput, nIO):
        cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes,
                          cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

    pred = np.argmax(bufferH[1])
    gt = label.item()
    num = pred == gt
    correct += num

    for b in bufferD:
        cudart.cudaFree(b)

acc = 100. * correct / len(test_dataset.dataset)
print("Succeeded running model in TensorRT!")
print(f'accuracy: {correct}/{len(test_dataset.dataset)} ({acc:.3f}%)')
end = time.time()
print(f"latency:{end - start}")
