import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataLoading import train_loader, test_loader


# Create the neural network
class KriaCNN(nn.Module):
    def __init__(self):
        super(KriaCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 64 * 64, 128)
        self.fc2 = nn.Linear(128, 2) 

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 64 * 64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Instantiate the model, define the loss function and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = KriaCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Training
def train(model_c, train_loader_c, criterion_c, optimizer_c, n_epochs):
    model_c.train()
    for epoch in range(n_epochs):
        epoch_loss = 0.0  # Initialize epoch loss
        for batch_idx, (data, target) in enumerate(train_loader_c):
            data, target = data.to(device), target.to(device)
            optimizer_c.zero_grad()
            output = model_c(data)
            loss = criterion_c(output, target)
            loss.backward()
            optimizer_c.step()
            # Accumulate batch loss
            epoch_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch + 1}, Batch {batch_idx}/{len(train_loader_c)}, Loss: {loss.item()}")

            # Print average epoch loss
        print(f"Epoch {epoch + 1}, Average Loss: {epoch_loss / len(train_loader_c)}")


def evaluate_model(model_c, _c):
    model_c.eval()
    correct = 0
    with torch.no_grad():
        for data, target in _c:
            data, target = data.to(device), target.to(device)
            output = model_c(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    print(f"Accuracy: {100. * correct / len(_c.dataset)}%")


def save_model(model_c, filepath):
    torch.save(model_c.state_dict(), filepath)


def main():
    train(model, train_loader, criterion, optimizer, 10)
    evaluate_model(model, test_loader)
    save_model(model, "./model.pth")  


if __name__ == "__main__":
    main()
