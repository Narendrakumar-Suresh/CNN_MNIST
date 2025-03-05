import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyperparameters
num_epochs=10
batch_size=64
learning_rate=0.0005

transform=transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))]
)

train_data=torchvision.datasets.MNIST(root='./data',train=True,download=True,transform=transform)
test_data=torchvision.datasets.MNIST(root='./data',train=False,download=True,transform=transform)

train_loader=torch.utils.data.DataLoader(train_data, batch_size=batch_size,shuffle=True)
test_loader=torch.utils.data.DataLoader(test_data, batch_size=batch_size,shuffle=False)

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1=nn.Conv2d(1,32,3)
        self.pool=nn.MaxPool2d(2,2)
        self.conv2=nn.Conv2d(32,16,3)

        self.fc1=nn.Linear(16*5*5,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,10)

    def forward(self,x):
        x=self.pool(F.relu(self.conv1(x)))
        x=self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)

        return x

model=CNN().to(device)
criterion=nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

n_steps=len(train_loader)

print('training started')
for epoch in range(num_epochs):
    for i,(image,sample) in enumerate(train_loader):
        image=image.to(device)
        sample=sample.to(device)

        output=model(image)
        loss=criterion(output,sample)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1)%2000==0:
            print(f'epoch{epoch+1}/{num_epochs}: loss: {loss.item():.4f}')

print('Training Finished!!!')

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for _ in range(10)]
    n_samples_correct = [0 for _ in range(10)]

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        probs = F.softmax(outputs, dim=1)  # Ensure proper probabilities
        _, predicted = torch.max(probs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        for i in range(len(labels)):
            label = labels[i].item()  # Get the label for the current sample
            pred = predicted[i].item()  # Get the prediction for the current sample

            if label == pred:
                n_class_correct[label] += 1
            n_samples_correct[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy is {acc}%')