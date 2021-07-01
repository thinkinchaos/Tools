import torch
from torchvision.models import resnet50
from thop import profile

# class YourModule(nn.Module):
#     # your definition
# def count_your_model(model, x, y):
#     # your rule here
#
# input = torch.randn(1, 3, 224, 224)
# flops, params = profile(model, inputs=(input, ),
#                         custom_ops={YourModule: count_your_model})

model = resnet50()
input = torch.randn(1, 3, 224, 224)
flops, params = profile(model, inputs=(input, ))


from thop import clever_format
macs, params = clever_format([flops, params], "%.3f")

print(flops, params)