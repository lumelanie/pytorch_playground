import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 20, bias=False)
        self.fc2 = nn.Linear(20, 10, bias=False)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def test_net(self):
        # test the net
        test_loss = 0
        correct = 0
        for data, target in testloader:
            data, target = Variable(data), Variable(target)
            data = data.view(-1, 28 * 28)
            net_out = self(data)
            # sum up batch loss
            test_loss += criterion(net_out, target).data.item()
            pred = net_out.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data).sum()
        test_loss /= len(testloader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct,
                                                                                     len(testloader.dataset),
                                                                                     100. * correct / len(
                                                                                         testloader.dataset)))
        acc = 100. * correct / len(testloader.dataset)
        return acc


def plot_weights(weights):
    vmin, vmax = np.min(weights), np.max(weights)
    fig = plt.figure(figsize=(10, 10))
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.2, hspace=0.2)
    for i in range(20):
        ax = fig.add_subplot(4, 5, i+1)
        i_weights = weights[i].reshape(28, 28)
        ax.matshow(i_weights, cmap='PiYG', vmin=vmin, vmax=vmax)
        ax.set_xticks(())
        ax.set_yticks(())
    plt.suptitle("accuracy: {}".format(acc.item()))
    plt.show()


if __name__ == "__main__":

    # load net
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    net.load_state_dict(torch.load('../nets/MNIST_MLP(20, 10).pt'))
    net.eval()

    # load data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.MNIST(root='../data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=4)
    testset = torchvision.datasets.MNIST(root='../data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=4)

    acc = net.test_net()
    weights = net.fc1.weight.data.numpy()
    plot_weights(weights)

    # modifiy net, test accuracy and plot weights
    sets = [[2, 13, 18], [3, 4, 12], [2, 13, 18, 3, 4, 12]]
    for set in sets:
        net.load_state_dict(torch.load('../nets/MNIST_MLP(20, 10).pt'))
        net.eval()
        for i in set:
            net.fc1.weight.data[i, :] = torch.from_numpy(np.zeros(784))
        weights = net.fc1.weight.data.numpy()
        acc = net.test_net()
        plot_weights(weights)

