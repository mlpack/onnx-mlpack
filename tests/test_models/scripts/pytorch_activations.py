#!/usr/bin/python3
#
# Generate a network with a whole bunch of different activations that can be
# mapped to mlpack.
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
    torch.nn.Linear(100, 25),
    torch.nn.CELU(alpha=1.02),
    torch.nn.Linear(25, 25),
    torch.nn.ELU(alpha=0.98),
    torch.nn.Linear(25, 25),
    torch.nn.GELU(approximate='none'),
    torch.nn.Linear(25, 25),
    torch.nn.GELU(approximate='tanh'),
    torch.nn.Linear(25, 25),
    torch.nn.Hardsigmoid(),
    torch.nn.Linear(25, 10),
    torch.nn.Hardswish(),
    torch.nn.Linear(10, 10),
    torch.nn.LeakyReLU(negative_slope=0.005),
    torch.nn.Linear(10, 10),
    torch.nn.Mish(),
    torch.nn.Linear(10, 10),
    torch.nn.Softplus(), # only default parameters supported
    torch.nn.Linear(10, 10),
    torch.nn.Tanh(),
    torch.nn.Linear(10, 3))
loss_fn = torch.nn.MSELoss(reduction='sum')

optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)
for t in range(100):
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    if t % 10 == 0:
        print(t, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

torch.onnx.export(model, (x,), "pytorch_activations.onnx", input_names=["x"],
    dynamo=False, external_data=False)
torch.onnx.export(model, (x,), "pytorch_activations_dynamo.onnx",
    input_names=["x"], dynamo=True, external_data=False)

# Compute outputs for a few inputs and store them for the tests.
y_pred = model(x[0:50, :])

x_np = x[0:50, :].detach().numpy()
y_pred_np = y_pred.detach().numpy()

np.savetxt("pytorch_activations_inputs.csv", x_np, delimiter=',')
np.savetxt("pytorch_activations_outputs.csv", y_pred_np, delimiter=',')
