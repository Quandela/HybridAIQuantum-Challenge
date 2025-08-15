import argparse
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import perceval as pcvl
import numpy as np
from boson_sampler import BosonSampler
from utils import MNIST_partial2, accuracy, plot_training_metrics
from merlin import QuantumLayer, OutputMappingStrategy

# DEPRECATED !! Now, we advise to install MerLin using pip install merlinquantum

def build_encoding_circuit(modes, hidden_dim):
    encoding_circuit = pcvl.Circuit(modes)
    for i in range(0, modes, 2):
        encoding_circuit.add(i, pcvl.BS())
    for i in range(1, modes - 1, 2):
        encoding_circuit.add(i, pcvl.BS())
    for i in range(hidden_dim):
        feat = pcvl.P(f"feat-{i + 1}")
        encoding_circuit.add(i % modes, pcvl.PS(feat))
    return encoding_circuit


def build_trainable_circuit(modes):
    return pcvl.GenericInterferometer(
        modes,
        lambda i: (pcvl.BS()
                   .add(0, pcvl.PS(pcvl.P(f"phase_train_1_{i}")))
                   .add(0, pcvl.BS())
                   .add(0, pcvl.PS(pcvl.P(f"phase_train_2_{i}"))))
    )


def train_model(model, optimizer, num_epochs, train_loader, val_loader, device):
    criterion = nn.CrossEntropyLoss()
    history_train_accuracy, history_val_accuracy, history_train_loss, history_val_loss = [], [], [], []

    for epoch in range(num_epochs):
        model.train()
        train_loss_epoch, train_acc_epoch = [], []

        for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss_epoch.append(loss.item())
            train_acc_epoch.append(accuracy(output, labels))

        model.eval()
        val_loss_epoch, val_acc_epoch = [], []
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validating"):
                images, labels = images.to(device), labels.to(device)
                output = model(images)
                loss = criterion(output, labels)
                val_loss_epoch.append(loss.item())
                val_acc_epoch.append(accuracy(output, labels))

        print(f"Epoch {epoch + 1}/{num_epochs} | "
              f"Train Loss: {sum(train_loss_epoch) / len(train_loss_epoch):.4f} | "
              f"Val Loss: {sum(val_loss_epoch) / len(val_loss_epoch):.4f} | "
              f"Train Acc: {sum(train_acc_epoch) / len(train_acc_epoch):.4f} | "
              f"Val Acc: {sum(val_acc_epoch) / len(val_acc_epoch):.4f}")

        history_train_loss.append(sum(train_loss_epoch) / len(train_loss_epoch))
        history_train_accuracy.append(sum(train_acc_epoch) / len(train_acc_epoch))
        history_val_loss.append(sum(val_loss_epoch) / len(val_loss_epoch))
        history_val_accuracy.append(sum(val_acc_epoch) / len(val_acc_epoch))

    plot_training_metrics(history_train_accuracy, history_val_accuracy, history_train_loss, history_val_loss)

def info_on_dataset(train_dataset, val_dataset):
    # print some information on the data
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    # Get data shape from first sample
    sample_data, _ = train_dataset[0]
    print(f"Data shape: {sample_data.shape}")
    train_labels = [train_dataset[i][1] for i in range(len(train_dataset))]
    val_labels = [val_dataset[i][1] for i in range(len(val_dataset))]

    # Get unique labels and counts
    train_unique, train_counts = np.unique(train_labels, return_counts=True)
    val_unique, val_counts = np.unique(val_labels, return_counts=True)

    print("Training set:")
    for label, count in zip(train_unique, train_counts):
        print(f"  Label {label}: {count}")

    print("Validation set:")
    for label, count in zip(val_unique, val_counts):
        print(f"  Label {label}: {count}")


def main(args):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    print(f"\n == Loading MNIST dataset ==")
    train_dataset = MNIST_partial2(split='train', digits = args.digits)
    val_dataset = MNIST_partial2(split='val', digits = args.digits)

    info_on_dataset(train_dataset, val_dataset)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    print(" == Data loaded == \n")

    print("\n == Building the model ==")
    print(" - From ResNet18 pretrained on ImageNet")
    pretrained_model = torchvision.models.resnet18(weights='IMAGENET1K_V1') if not args.random else torchvision.models.resnet18()
    print(" - Changing the first convolution to match MNIST unique channel")
    pretrained_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    if args.quantum:
        print("\n - Building quantum circuit with Perceval")
        encoding_circuit = build_encoding_circuit(args.modes, args.hidden_dim)
        trainable_circuit = build_trainable_circuit(args.modes)

        circuit = pcvl.Circuit(args.modes)
        circuit.add(0, encoding_circuit, merge=True)
        circuit.add(0, trainable_circuit, merge=True)
        input_state = [(i + 1) % 2 for i in range(args.modes)]

        print(" - Merging this circuit as a TorchModule with MerLin")
        qlayer = QuantumLayer(
            input_size=args.hidden_dim,
            output_size=len(args.digits),
            circuit=circuit,
            input_state=input_state,
            trainable_parameters=[p.name for p in circuit.get_parameters() if not p.name.startswith("feat")],
            input_parameters = ["feat"],
            output_mapping_strategy=OutputMappingStrategy.LINEAR
        )

        class DivideByPi(nn.Module):
            def forward(self, x):
                return x / torch.pi
        print(" - Quantum circuit initialised !")
        print(" - Changing the last FC layer")
        pretrained_model.fc = nn.Sequential(
            nn.Linear(512, args.hidden_dim),
            torch.nn.Sigmoid(),
            DivideByPi(),
            qlayer
        )
    else:
        print("\n - Building a classical classification head")

        pretrained_model.fc = nn.Sequential(
            nn.Linear(512, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, len(args.digits)),
        )


    model = pretrained_model.to(device)
    model.requires_grad_(False)
    model.fc.requires_grad_(True)
    optimizer = torch.optim.Adam(
        [p for p in model.fc.parameters() if p.requires_grad], lr=args.lr)

    print("\n == MODEL TRAINING ==")
    train_model(model, optimizer, args.epochs, train_loader, val_loader, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a QNN ResNet model with Perceval quantum layers")
    parser.add_argument('--modes', type=int, default=8, help='Number of modes for quantum circuit')
    parser.add_argument('--hidden-dim', type=int, default=128, help='Hidden dimension before QLayer')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--digits', nargs='+', type=int, default=[2,5], help='List of digits to classify (e.g., 2 6 7)')
    parser.add_argument('-quant', '--quantum', action='store_true', default=False, help='Set if we use Quantum TL')
    parser.add_argument( '--random', action='store_true', default=False, help='Set if we use a Random Encoder')

    args = parser.parse_args()
    main(args)
