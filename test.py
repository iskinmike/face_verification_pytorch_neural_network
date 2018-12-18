# -*- coding: utf-8 -*-
import torch
import matplotlib.pyplot as plt
from ruamel.yaml import YAML
import sys
import load_test as tloader
import numpy as np
import os


#===========================
# ========================= support functions
def create_model(D_in, D_out, learning_rate):
    hidden_layer_1_in = D_in;
    hidden_layer_1_out = 5;
    # hidden_layer_2_in = 10000;
    # hidden_layer_2_out = 1000;
    # hidden_layer_3_in = 1000;
    # hidden_layer_3_out = 100;
    # hidden_layer_4_in = 10000;
    # hidden_layer_4_out = 10;
    hidden_layer_5_in = 5;
    hidden_layer_5_out = D_out;

    # Use the nn package to define our model and loss function.
    model = torch.nn.Sequential(
        torch.nn.Linear(hidden_layer_1_in, hidden_layer_1_out),
        torch.nn.ReLU(),
        # torch.nn.Linear(hidden_layer_3_in, hidden_layer_3_out),
        # torch.nn.ReLU(),
        torch.nn.Linear(hidden_layer_5_in, hidden_layer_5_out),
    )
    loss_fn = torch.nn.MSELoss(reduction='sum')
    print("model created")

    # Use the optim package to define an Optimizer that will update the weights of
    # the model for us. Here we will use Adam; the optim package contains many other
    # optimization algoriths. The first argument to the Adam constructor tells the
    # optimizer which Tensors it should update.
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return model, loss_fn, optimizer


def learn_to_success(model, input, output, loss_fn, optimizer, learning_rate):
    print("Start learning")
    epoch = 1;
    max_epoch = 30000;
    loss_res = learning_rate + 1;
    while (loss_res> learning_rate or epoch < max_epoch):
    # for t in range(epochs):
        # Forward pass: compute predicted y by passing x to the model.
        y_pred = model(input)

        # Compute and print loss.
        loss = loss_fn(y_pred, output)
        if (epoch) % 5 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch, max_epoch, loss.item()))
        # print(t, loss.item())
        loss_res = loss.item()

        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the variables it will update (which are the learnable
        # weights of the model). This is because by default, gradients are
        # accumulated in buffers( i.e, not overwritten) whenever .backward()
        # is called. Checkout docs of torch.autograd.backward for more details.
        optimizer.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model
        # parameters
        loss.backward()

        # Calling the step function on an Optimizer makes an update to its
        # parameters
        optimizer.step()
        epoch += 1
    print("Model learned")
    pass

def create_model_example(D_in, D_out, learning_rate):
    layer_1_in = D_in;
    layer_1_out = 5;
    layer_2_in = layer_1_out;
    layer_2_out = D_out;

    # Use the nn package to define our model and loss function.
    model = torch.nn.Sequential(
        torch.nn.Linear(layer_1_in, layer_1_out),
        torch.nn.ReLU(),
        torch.nn.Linear(layer_2_in, layer_2_out),
    )
    loss_fn = torch.nn.MSELoss(reduction='sum')

    # Use the optim package to define an Optimizer that will update the weights of
    # the model for us.
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return model, loss_fn, optimizer
    pass

def learn_model_example(model, input, output, loss_fn, optimizer, learning_rate):
    print("Start learning")
    epoch = 1;
    max_epoch = 30000;
    loss_res = learning_rate + 1;
    while (loss_res > learning_rate or epoch < max_epoch):
        # Forward pass: compute predicted y by passing x to the model.
        y_pred = model(input)

        # Compute and print loss.
        loss = loss_fn(y_pred, output)
        loss_res = loss.item()

        optimizer.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model
        # parameters
        loss.backward()

        # Calling the step function on an Optimizer makes an update to its
        # parameters
        optimizer.step()
        epoch += 1
    print("Model learned")
    pass


def learn_model(model, input, output, loss_fn, optimizer, epochs):
    print("Start learning")
    for t in range(epochs):
        # Forward pass: compute predicted y by passing x to the model.
        y_pred = model(input)

        # Compute and print loss.
        loss = loss_fn(y_pred, output)
        if (t + 1) % 5 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(t + 1, epochs, loss.item()))
        # print(t, loss.item())

        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the variables it will update (which are the learnable
        # weights of the model). This is because by default, gradients are
        # accumulated in buffers( i.e, not overwritten) whenever .backward()
        # is called. Checkout docs of torch.autograd.backward for more details.
        optimizer.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model
        # parameters
        loss.backward()

        # Calling the step function on an Optimizer makes an update to its
        # parameters
        optimizer.step()
    print("Model learned")
    pass
#===========================



# Load data
input_data_path = "input.dat"
result_data_path = "result.dat"

inp_exists = os.path.isfile(input_data_path)
res_exists = os.path.isfile(result_data_path)

x, y = tloader.load_training_data(input_data_path, result_data_path)

D_out = 1
D_in = x.size()[1]
learning_rate = 1e-4
model, loss_fn, optimizer = create_model(D_in, D_out, learning_rate)

model_data_path = "model.dat"
model_exists = os.path.isfile(model_data_path)
if model_exists:
    model.load_state_dict(torch.load(model_data_path))
    model.eval()
    pass
else:
    # learn_model(model, x, y, loss_fn, optimizer, 2000)
    learn_to_success(model, x, y, loss_fn, optimizer, learning_rate)
    torch.save(model.state_dict(), model_data_path)
    pass


test_samples_data_path = "test_samples.dat"
test_samples = tloader.load_sample_data(test_samples_data_path)

print("Predict samples")
predicted = model(test_samples).detach().numpy()

print(predicted)

pred_path = "predictions.dat"
torch.save(predicted, pred_path)



