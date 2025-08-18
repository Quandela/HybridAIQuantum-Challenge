import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader
from utils import MNIST_partial

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

class SharedWeightFC(nn.Module):
    def __init__(self, in_features, out_features, shared_rows):
        super(SharedWeightFC, self).__init__()
        self.shared_weights = nn.Parameter(torch.randn(shared_rows, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        self.out_features = out_features
        self.shared_rows = shared_rows

    def forward(self, x):
        weight_matrix = self.shared_weights.repeat(self.out_features // self.shared_rows, 1)
        return torch.matmul(x, weight_matrix.t()) + self.bias

class CNNModel(nn.Module):
    def __init__(self, use_weight_sharing=False, shared_rows=10):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(8, 12, kernel_size=5)
        
        if use_weight_sharing:
            self.fc1 = SharedWeightFC(in_features=12*4*4, out_features=20, shared_rows=shared_rows)
            self.fc2 = nn.Linear(20, 10)
        else:
            self.fc1 = nn.Linear(12*4*4, 20)
            self.fc2 = nn.Linear(20, 10)
        
    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

def apply_pruning(model, amount=0.3):
    """Apply structured pruning to convolutional and fully connected layers."""
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            prune.ln_structured(layer, name='weight', amount=amount, n=2, dim=0)
            print(f"Applied structured pruning to Conv2d layer {name} with {amount * 100:.1f}% filters pruned.")
        elif isinstance(layer, nn.Linear):
            prune.ln_structured(layer, name='weight', amount=amount, n=2, dim=1)
            print(f"Applied structured pruning to Linear layer {name} with {amount * 100:.1f}% neurons pruned.")

def remove_pruning(model):
    """Remove pruning masks to finalize the reduced model."""
    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            try:
                prune.remove(layer, 'weight')
            except ValueError:
                pass  # Layer was not pruned

class MaskedAdam(torch.optim.Adam):
    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)

    def step(self, closure=None):
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue
                grad_mask = param.data != 0
                param.grad.data.mul_(grad_mask)
        super().step(closure)

def create_datasets():
    train_dataset = MNIST_partial(split='train')
    val_dataset = MNIST_partial(split='val')
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
    return train_dataset, val_dataset, train_loader, val_loader, batch_size

def train_classical_cnn(train_loader, val_loader, num_epochs, use_pruning=False, pruning_amount=0.5, use_weight_sharing=False, shared_rows=10):
    learning_rate = 1e-3
    
    model = CNNModel(use_weight_sharing=use_weight_sharing, shared_rows=shared_rows)
    
    if use_pruning:
        apply_pruning(model, amount=pruning_amount)
        remove_pruning(model)
        
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    if use_pruning:
        optimizer = MaskedAdam(model.parameters(), lr=learning_rate)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    num_classical_parameter = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("# of parameters in classical CNN model: ", num_classical_parameter)
    
    for epoch in range(num_epochs):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            if (i+1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
    
    model.eval()
    correct = 0
    total = 0
    loss_test_list = [] 
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss_test = criterion(outputs, labels).cpu().detach().numpy()
            loss_test_list.append(loss_test)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f"Accuracy on the test set: {(100 * correct / total):.2f}%")
    return model