import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np


# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 128)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(128, 10)  # Output layer

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the input
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        out = self.fc4(x)
        return out


# Training function
def train(model, device, train_loader, optimizer):
    model.train()
    criterion = nn.CrossEntropyLoss()
    for data, target in train_loader:
        optimizer.zero_grad()
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


# Evaluation function to collect response patterns
def evaluate(model, device, calibrate_loader):
    model.eval()
    response_patterns = []
    targets_list = []
    with torch.no_grad():
        for data, target in calibrate_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            preds = output.argmax(dim=1)
            response_patterns.extend(preds.cpu().numpy())
            targets_list.extend(target.cpu().numpy())
    return np.array(response_patterns), np.array(targets_list)


if __name__ == "__main__":
    np.random.seed(1)
    # Main function to run multiple training sessions and collect data
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    num_runs = 40       # Number of training runs
    num_epochs = 15    # Number of epochs per run
    batch_size = 1000    # Batch size for training

    # Data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Load MNIST dataset
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    calibrate_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    # Data loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    calibrate_loader = DataLoader(dataset=calibrate_dataset, batch_size=1000, shuffle=False)

    # Store all response patterns
    all_response_patterns = []

    learning_rate = np.linspace(0.01, 0.5, num_runs).tolist()
    
    # Run multiple training sessions
    for run in range(num_runs):
        # lr = np.random.uniform(0.1, 3) * 0.1
        # print(lr)
        print(f"\nStarting training run {run + 1}/{num_runs}")
        model = SimpleNN().to(device)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate[run])

        for epoch in range(1, num_epochs + 1):
            train(model, device, train_loader, optimizer)
            responses, targets = evaluate(model, device, calibrate_loader)

            # Print progress
            correct = np.sum(responses == targets)
            total = len(targets)
            accuracy = 100.0 * correct / total
            print(f"Run {run + 1}, Epoch {epoch}: Accuracy on calibration set: {accuracy:.2f}%")

            # Collect response patterns for calibration
            all_response_patterns.append({
                'run': run,
                'epoch': epoch,
                'responses': responses,
                'targets': targets,
                'correct': correct,
                'accuracy': accuracy
            })

    # The collected data can now be used for IRT calibration
    with open('response_patterns.npy', 'wb') as f:
        np.save(f, all_response_patterns)
