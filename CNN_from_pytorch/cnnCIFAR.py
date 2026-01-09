#By Michael Stull
#Last Updated 12/22/2025
#Code Sources
#   Pytorch Tutorial for LeNet Architecture
#   (brother)Kevin Stull for Consulting on Training Engineering
#   Model Load Function is from Claude
#Learned Techniques:
#   1. Dropout
#   2. Cut and Mix//Affine Transformations and Rotation for overfitting prevention
#   3. Model Normalization
#   4. Validation vs Test Data and 'Test Hacking' by Using Test Data to Validate
#       - In this case I used test data for validation because I was working on this on a plane and was more focused on implementing other efficiencies
#       - So technically there is no real test data since I am using my Test Set for validation
#   5. Hyper Parameter Tuning, Comparing the Validation Loss to the Training Loss
#   6. Tqdm Loading bar Makes training less tedious and more digestable especially with less compute(Laptop: 100epochs = 45minutes... no manches)
#   6.5 Set up ssh on a more powerful computer so that it doesnt take so long.
#   7. Evaluate the data set for possible bias or non-normal distribution of data, in this case every value hass exactly 10,000 examples but this is not always true
#   8. The Goal is generlization 
#Higher Precision and Important Remeberances
#   1. Larger Model, LeNet is only approx. 60,000 parameters, Larger Models will perform better. I didd this on a laptop without an NVIDIA Gpu or internet
#   2. Proper Validation Splitting, validation should ONLY be done from the training data, everytime you look at the Test data you make your findings less significant. 
#       -Test Data: "Lock it and throw away the key", "Anytime you use Test Data it should be reported in the paper"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm,trange

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
num_epochs = int(input("Desired number of Epochs:"))
batch_size = 16
learning_rate = 0.001
PATH = './cnn.pth'

# dataset has PILImage images of range [0, 1]. 
# We transform them to Tensors of normalized range [-1, 1]
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=2), # Rotates between -30 and +30 degrees
    transforms.RandomAffine(
        degrees=0, 
        translate=(0.1, 0.1), # Max 10% horizontal and vertical translation
        scale=(0.8, 1.2),     # Scale between 80% and 120%
        shear=10              # Shear between -10 and +10 degrees
    ),
    transforms.ToTensor(), # Converts the image to a PyTorch tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
])
test_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
# CIFAR10: 60000 32x32 color images in 10 classes, with 6000 images per class
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=train_transform)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=test_transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                          shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                         shuffle=False)


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.dropout = nn.Dropout(p=0.1)
        self.dropout2d = nn.Dropout2d(p=0.1)
        self.conv1 = nn.utils.weight_norm(nn.Conv2d(3, 6, 5))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.utils.weight_norm(nn.Conv2d(6, 16, 5))
        self.fc1 = nn.utils.weight_norm(nn.Linear(16 * 5 * 5, 120))
        self.fc2 = nn.utils.weight_norm(nn.Linear(120, 84))
        self.fc3 = nn.utils.weight_norm(nn.Linear(84, 10))

    def forward(self, x):
        # -> n, 3, 32, 32
        x = self.pool(F.relu(self.conv1(x)))  # -> n, 6, 14, 14
        x = self.dropout2d(x)
        x = self.pool(F.relu(self.conv2(x)))  # -> n, 16, 5, 5
        x = self.dropout2d(x)
        x = x.view(-1, 16 * 5 * 5)            # -> n, 400
        x = F.relu(self.fc1(x)) 
        x = self.dropout(x)              # -> n, 120
        x = F.relu(self.fc2(x))
        x = self.dropout(x)               # -> n, 84
        x = self.fc3(x)                       # -> n, 10
        return x

def load_model(path='./cnn.pth'):
    """Load the saved model from disk"""
    model = ConvNet()
    
    # Load the state dict
    state_dict = torch.load(path)
    
    # Check if weight normalization was used (has weight_g and weight_v)
    if any('weight_g' in key for key in state_dict.keys()):
        print("Detected weight normalization in saved model. Applying weight_norm...")
        # Apply weight normalization to match the saved model structure
        model.conv1 = model.conv1
        model.conv2 = model.conv2
        model.fc1 = model.fc1
        model.fc2 = model.fc2
        model.fc3 = model.fc3
    
    model.load_state_dict(state_dict)
    return model


load = input("Load Model y or n")[0]=='y'
if load =='Y' or load == 'y':
    model = load_model('./cnn.pth')
else:
    model = ConvNet()


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_loss = np.zeros(num_epochs)
validation_loss = np.zeros(num_epochs)
n_total_steps = len(train_loader)

for epoch in range(num_epochs):
    pbar = tqdm(train_loader, total=n_total_steps, 
                desc=f'Epoch {epoch+1}/{num_epochs}')
    loss_2 = None
    loss_delta=0
    model.train()
    for i, (images, labels) in enumerate(pbar):
        
        # origin shape: [4, 3, 32, 32] = 4, 3, 1024
        # input_layer: 3 input channels, 6 output channels, 5 kernel size
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss[epoch] += loss.item()
        
        loss_1 = loss.item()
        if loss_2 is not None:
            loss_delta += loss_1 - loss_2
        else:
            loss_delta = 0
        loss_2 = loss_1
        pbar.set_postfix({'loss_delta': f'{loss_delta:.4f}'})
    train_loss[epoch] = train_loss[epoch]/(len(train_loader))

    #validation split
    model.eval()
    with torch.no_grad():
        for images,labels in test_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            validation_loss[epoch] +=loss
        validation_loss[epoch] = validation_loss[epoch]/(len(test_loader))

print('Finished Training')

model.eval()
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
        
        for i in range(len(labels)):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')
    acc= np.zeros(10)
    for i in range(10):
        acc[i] = 100.0 * n_class_correct[i] / n_class_samples[i]

plt.bar(classes,acc,color='blue')
plt.xlabel('Classes')
plt.ylabel('Accuraccy')
plt.title('Validation')
plt.show()

plt.plot(range(len(train_loss)),train_loss,color='red',label="Train")
plt.plot(range(len(validation_loss)),validation_loss,color='green',label="Validation")
plt.xlabel("Epoch Step")
plt.ylabel("Loss")
plt.legend()
plt.show()

save = input("Save Model y or n")[0]=='y'
if save == 'Y' or save == 'y':
    torch.save(model.state_dict(), PATH)
    print("MODEL SAVED at epoch:",epoch)
else:
    print("MODEL Not Saved")