import torch
import torch.nn.functional as F
from util_test import data_compare

input=torch.load("./layer_norm1_intput.pt")
normalized_shape=(1152,)
weight=torch.load("./layer_norm1_weight.pt")
bias=torch.load("./layer_norm1_bias.pt")
output_npu=torch.load("./layer_norm1_output.pt")
# print("output_npu:", torch.max(output_npu))
eps = 1e-06

# output_gpu=F.layer_norm(input, normalized_shape, weight, bias, eps).to('cuda')
y=F.layer_norm(input.to('cpu'), normalized_shape, weight.to('cpu'), bias.to('cpu'), eps)

y1=F.layer_norm(input, normalized_shape, weight, bias, eps)
print(bias)
# print("output_gpu_cpu", torch.max(output_gpu_cpu))
# print("output_gpu_cpu dtype", output_gpu_cpu.dtype)

print("cpu dtype is ",y.dtype)
print("npu dtype is ",y1.dtype)
print("================= y =====================")
data_compare(y1.to(torch.float32).detach().cpu().numpy(), y.to(torch.float32).detach().cpu().numpy(), diff_thd=0.001, pct_thd=0.001)
# print("================= mean =====================")
# data_compare(mean1.detach().cpu().numpy(), mean.cpu().numpy(), diff_thd=0.001, pct_thd=0.001)
# print("================= rstd =====================")
# data_compare(rstd1.detach().cpu().numpy(), rstd.cpu().numpy(), diff_thd=0.001, pct_thd=0.001)