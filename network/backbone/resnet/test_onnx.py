import torch

from torchsummary import summary
from zmq import device
from resnet import resnet50, resnet34, resnet18

USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('CUDA 사용 확인:', device)
model = resnet34().to(device)

dummy_input = torch.randn(10, 3, 224, 224, device='cuda')
dummy_output = model(dummy_input)

torch.onnx.export(model, dummy_input, "resnet34.onnx", verbose=True, example_outputs = dummy_output)

