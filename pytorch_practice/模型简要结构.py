
# %% 使用torchsummary查看
import torchvision.models as models
from torchsummary import summary
import torch
from torch import nn
 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vgg = models.vgg19().to(device)
 
summary(vgg, (3, 224, 224))

# %% 使用netron查看
import torch
from simple_mltiple_classification import Net
import netron

net = Net()
x = torch.rand(1, 3, 32, 32)
path = "./temp_result.pth"
torch.onnx.export(net, x, path)
netron.start(path)




# %%
