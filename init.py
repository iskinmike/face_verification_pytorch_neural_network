import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable

import numpy as np

class Net(nn.Module):
    conv1_in_channels = 1
    conv1_out_channels = 80
    conv2_in_channels = conv1_out_channels
    conv2_out_channels = 120
    conv_kernel = 2
    fc1_out_channels = 120
    fc2_out_channels = 84
    fc3_out_channels = 10
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv1d(self.conv1_in_channels, self.conv1_out_channels, self.conv_kernel)
        self.conv2 = nn.Conv1d(self.conv2_in_channels, self.conv2_out_channels, self.conv_kernel)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(self.conv2_out_channels, self.fc1_out_channels)
        self.fc2 = nn.Linear(self.fc1_out_channels, self.fc2_out_channels)
        self.fc3 = nn.Linear(self.fc2_out_channels, self.fc3_out_channels)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool1d(F.relu(self.conv1(x)), 2)
        # If the size is a square you can only specify a single number
        x = F.max_pool1d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


learning_rate = 0.001
num_epochs = 25

net = Net()
print(net)

# create data

input_size = 20
hidden_size = 50
output_size = 7
n_layers = 2

batch_size = 80
seq_len = 20

core_size = 5
conv1_in_channels = 1
conv1_out_channels = 10


x_train = Variable(torch.rand(seq_len, conv1_in_channels, conv1_out_channels)) #torch.rand(2, 50, 1)
y_train = torch.rand(seq_len, 10)


criterion = nn.MSELoss() #torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    # Convert numpy arrays to torch tensors
    inputs = x_train
    targets = y_train

    # Forward pass
    outputs = net(inputs)
    loss = criterion(outputs, targets)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 5 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))

# Plot the graph
test = torch.rand(14, conv1_in_channels, conv1_out_channels)

predicted = net(test).detach().numpy()
# plt.plot(test, y_train, 'ro', label='Original data')

# test = torch.rand(2, 2, 3)
x_data = test.numpy()
# print(x_data)
# print("size: ")
# print(x_data.size)
x_data = x_data[:, 0, :]
# print(x_data)
# print("size: ")
# print(x_data.size)

np.newaxis

plt.plot(x_data, predicted, label='Fitted line')
plt.legend()
plt.show()
