import argparse
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import perceval as pcvl

from boson_sampler import BosonSampler
from utils import MNIST_partial2, accuracy, plot_training_metrics
from pqnn_model import QuantumLayer, OutputMappingStrategy


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


def main(args):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    train_dataset = MNIST_partial2(split='train')
    val_dataset = MNIST_partial2(split='val')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    pretrained_model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
    pretrained_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    encoding_circuit = build_encoding_circuit(args.modes, args.hidden_dim)
    trainable_circuit = build_trainable_circuit(args.modes)

    circuit = pcvl.Circuit(args.modes)
    circuit.add(0, encoding_circuit, merge=True)
    circuit.add(0, trainable_circuit, merge=True)

    input_state = [(i + 1) % 2 for i in range(args.modes)]

    qlayer = QuantumLayer(
        input_size=args.hidden_dim,
        output_size=2,
        circuit=circuit,
        input_state=input_state,
        trainable_parameters=[p.name for p in circuit.get_parameters() if not p.name.startswith("feat")],
        output_mapping_strategy=OutputMappingStrategy.LINEAR
    )

    pretrained_model.fc = nn.Sequential(
        nn.Linear(512, args.hidden_dim),
        qlayer
    )

    model = pretrained_model.to(device)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr)

    train_model(model, optimizer, args.epochs, train_loader, val_loader, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a QNN ResNet model with Perceval quantum layers")
    parser.add_argument('--modes', type=int, default=8, help='Number of modes for quantum circuit')
    parser.add_argument('--hidden-dim', type=int, default=128, help='Hidden dimension before QLayer')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')

    args = parser.parse_args()
    main(args)
