import torch

from torchsummary import summary
from zmq import device
from resnet import resnet50, resnet34, resnet18

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = resnet50().to(device)

dummy_input = torch.randn(10, 3, 224, 224, device='cuda')
dummy_output = model(dummy_input)

torch.onnx.export(model, dummy_input, "resnet34.onnx", verbose=True, example_outputs = dummy_output)

