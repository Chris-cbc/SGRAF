path = "F:/SGRAF/runs/f30k_SGR/checkpoint/model_best.pth.tar"
import torch
from model import SGRAF
import torchvision.models as models

checkpoint = torch.load(path)  # pytorch模型加载
opt = checkpoint['opt']
model = SGRAF(opt)
model.load_state_dict(checkpoint['model'])

batch_size = 1  # 批处理大小
shape1 = (480, 1024)  # 输入数据,改成自己的输入shape
shape2 = (300, 1024)  # 输入数据,改成自己的输入shape
shape3 = (512, 1024)  # 输入数据,改成自己的输入shape

x1 = torch.randn(batch_size, *shape1)  # 生成张量
x2 = torch.randn(batch_size, *shape2)  # 生成张量
x3 = torch.randn(batch_size, *shape3)  # 生成张量
shapes = (x1, x2, x3)

export_onnx_file = "test.onnx"  # 目的ONNX文件名
torch.onnx.export(model, shapes,
                  export_onnx_file,
                  do_constant_folding=True,  # 是否执行常量折叠优化
                  input_names=["input"])
