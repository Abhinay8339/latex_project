import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import time

# Load data
transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
train_loader = torch.utils.data.DataLoader(datasets.MNIST('.', train=True, download=True, transform=transform), batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(datasets.MNIST('.', train=False, transform=transform), batch_size=1000)

# Define model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(784, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train
start = time.time()
for epoch in range(5):
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
end = time.time()
print(f"Training time: {end - start:.2f} seconds")

# Evaluate
correct = 0
total = 0
model.eval()
with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
print(f"Test accuracy: {100 * correct / total:.2f}%")

# Export to ONNX
dummy_input = torch.randn(1, 784)
torch.onnx.export(model, dummy_input, "model.onnx", input_names=["input"], output_names=["output"])
print("Exported to model.onnx")

