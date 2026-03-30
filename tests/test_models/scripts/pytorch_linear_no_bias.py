#!/usr/bin/python3
#
# Generate a 3-layer Linear network with PyTorch, and export to ONNX.
#
# This is based on the example here:
# https://docs.pytorch.org/tutorials/beginner/pytorch_with_examples.html#pytorch-nn

import torch
import math
import numpy as np

x = torch.randn((2000, 100)) + \
    torch.linspace(-math.pi, math.pi, 2000).unsqueeze(-1)
y = 0.05 * torch.randn((2000, 3)) + \
    torch.sum(torch.sin(x)).unsqueeze(-1)

model = torch.nn.Sequential(
    torch.nn.Linear(100, 25, bias=False),
    torch.nn.Linear(25, 10, bias=False),
    torch.nn.Linear(10, 3, bias=False))
loss_fn = torch.nn.MSELoss(reduction='sum')

optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)
for t in range(10000):
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    if t % 10 == 0:
        print(t, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

torch.onnx.export(model, (x,), "pytorch_linear_no_bias.onnx", input_names=["x"],
    dynamo=False, external_data=False)
torch.onnx.export(model, (x,), "pytorch_linear_no_bias_dynamo.onnx", input_names=["x"],
    dynamo=True, external_data=False)

# Compute outputs for a few inputs and store them for the tests.
y_pred = model(x[0:50, :])

x_np = x[0:50, :].detach().numpy()
y_pred_np = y_pred.detach().numpy()

np.savetxt("pytorch_linear_no_bias_inputs.csv", x_np, delimiter=',')
np.savetxt("pytorch_linear_no_bias_outputs.csv", y_pred_np, delimiter=',')
