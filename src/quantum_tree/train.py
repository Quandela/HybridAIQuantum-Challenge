from model import QNN, CNNBaseline
from boson_sampler import BosonSampler
import utils
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")
_ = torch.manual_seed(1618)
torch.set_float32_matmul_precision('high')

parser = argparse.ArgumentParser(
    description="Train a QNN or CNN baseline model on a partial MNIST dataset with quantum neural network or classical convolutional network."
)
parser.add_argument("--m", type=int, default=20, 
                    help="Number of qumodes; use 12 if testing on a real QPU.")
parser.add_argument("--n", type=int, default=3, 
                    help="Number of parameters (e.g., circuit depth or layers).")
parser.add_argument("--cnn", action="store_false", 
                    help="Use CNNBaseline instead of QNN.")
parser.add_argument("--batch_size", type=int, default=256, 
                    help="Training batch size.")
parser.add_argument("--lr", type=float, default=2e-3, 
                    help="Learning rate for the optimizer.")
parser.add_argument("--weight_decay", type=float, default=1e-3, 
                    help="Weight decay (L2 regularization factor).")
parser.add_argument("--epochs", type=int, default=50, 
                    help="Number of training epochs.")
parser.add_argument("--label_smoothing", type=float, default=0.1, 
                    help="Label smoothing factor for the cross-entropy loss.")
args = parser.parse_args()

m = args.m
n = args.n
qnn = args.cnn
batch_size = args.batch_size
lr = args.lr
weight_decay = args.weight_decay
epochs = args.epochs
label_smoothing = args.label_smoothing
save_path = "qnn.pkl" if qnn else "cnn.pkl"

train_dataset = utils.MNIST_partial(split = 'train')
val_dataset = utils.MNIST_partial(split='val')

train_loader = DataLoader(train_dataset, batch_size, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size, shuffle = False)

print(f"Train size: {len(train_loader)*batch_size}, Val size: {len(val_loader)*batch_size}")

if qnn:
    bs = BosonSampler(m, n)
    print("embed size:", bs.embedding_size)
    print("gate angles:", bs.nb_parameters)
    print("number of qumodes:", bs.m)
    model = QNN(bs, device).to(device)
else:
    model = CNNBaseline().to(device)

try:
    print("Number of parameters:", sum([param.numel() for param in model.parameters() if param.requires_grad]) - sum([param.numel() for param in model.surrogate.parameters() if param.requires_grad]))
except:
    print("Number of parameters:", sum([param.numel() for param in model.parameters() if param.requires_grad]))

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=2e-6)

print("Starting Training...")

losses = []
accuracies = []

train_loss, train_acc, val_loss, val_acc = [], [], [], []
surrogate_rate = 5.0
for epoch in range(epochs):
    training_losses, training_accs, total = 0, 0, 0
    model.train()

    for step, (batch, gt) in enumerate(train_loader):
        optimizer.zero_grad()
        batch, gt = batch.to(device), gt.to(device)

        if qnn:
            output, l, real = model(batch)
            loss = loss_fn(output, gt)*0.5 + l*surrogate_rate + loss_fn(real, gt)*0.25
        else:
            output = model(batch)
            loss = loss_fn(output, gt)

        loss.backward() 
        optimizer.step()
        training_losses += loss.item()
        
        if qnn:
            _, predicted = torch.max(real, dim=-1)
        else:
            _, predicted = torch.max(output, dim=-1)

        training_accs += (predicted == gt).sum().item()
        total += gt.size(0)
        losses.append(loss.item())
        accuracies.append(training_accs/total)

        if qnn:
            print(f"Iteration {step+1}/{len(train_loader)}, Loss = {loss.item():.6f}, Acc = {training_accs/total*100:.3f}%")
    
    surrogate_rate *= 0.95
    
    validation_loss, validation_acc = utils.evaluate(model, val_loader, device)

    training_losses = training_losses/len(train_loader)
    training_accs = training_accs/total*100
    train_loss.append(training_losses)
    train_acc.append(training_accs)
    val_loss.append(validation_loss)
    val_acc.append(validation_acc)
    
    scheduler.step()
    print(f"Epoch {epoch+1}, Train Acc = {training_accs:.3f}%, Train Loss = {training_losses:.6f}, Eval Acc = {validation_acc:.3f}%, Eval Loss = {validation_loss:.6f}")

with open(save_path, 'wb') as f:
    pickle.dump((losses, accuracies, train_acc, val_acc, train_loss, val_loss, "QNN" if qnn else "CNN", model.state_dict(), str(model)), f)

print("Saved")

utils.plot_training_metrics_detailed([save_path])

print("Finished training")