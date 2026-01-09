import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# 1. Define transformations
# Converts the images into PyTorch tensors and normalizes them


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 2. Load the training dataset
train_dataset = torchvision.datasets.MNIST(
    root='./data',        # Directory to save the data
    train=True,           # Specify training data
    download=True,        # Downloads the data if not present
    transform=transform
)

# 3. Load the testing dataset
test_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=False,          # Specify test data
    download=True,
    transform=transform
)

# 4. Create DataLoaders for batching and shuffling
BATCH_SIZE = 32 # Example batch size

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    BATCH_SIZE,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    BATCH_SIZE,
    shuffle=False
)

classes = ('0', '1', '2', '3',
           '4', '5', '6', '7', '8', '9')

def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    print(img.size())
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(train_loader)
images, labels = next(dataiter)



class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # -> n, 3, 32, 32
        x = self.pool(F.relu(self.conv1(x)))  # -> n, 1, 28, 28
        x = self.pool(F.relu(self.conv2(x)))  # -> n, 16, 5, 5
        x = x.view(-1, 16 * 4 * 4)  
                  # -> n, 256
        x = F.relu(self.fc1(x))               # -> n, 120
        x = F.relu(self.fc2(x))               # -> n, 84
        x = self.fc3(x)                       # -> n, 10
        return x
    

model = ConvNet()
learning_rate = .001
num_epochs = 15

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr= learning_rate)

n_total_steps = len(train_loader)

for epoch in range(num_epochs):
    loss_1,loss_2,delta =0,0,0

    for i,(images,labels) in enumerate(train_loader):
        outputs = model(images)
        loss = criterion(outputs,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_1 = loss
        if not(loss_2 == 0):
            delta += (loss_2-loss_1).item()
        loss_2 = loss
    print('delta:',delta)

print('Finished Training')
PATH = './cnn.pth'
torch.save(model.state_dict,PATH)

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for (images,labels) in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
        
        for i in range(labels.size(0)):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')

    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc} %')