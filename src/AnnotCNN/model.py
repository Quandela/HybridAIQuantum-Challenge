import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import accuracy

class MnistModel(nn.Module):
    def __init__(self, device="cpu", embedding_size=0):
        """
        A simple MNIST classifier.
        If embedding_size > 0, the model expects an additional quantum embedding vector to be concatenated with the flattened image.
        """
        super().__init__()
        input_size = 28 * 28
        if embedding_size:
            input_size += embedding_size
        self.device = device
        self.embedding_size = embedding_size
        self.linear = nn.Linear(input_size, 10)

    def forward(self, xb, emb=None):
        xb = xb.view(-1, 784)
        if self.embedding_size and emb is not None:
            xb = torch.cat((xb, emb), dim=1)
        out = self.linear(xb)
        return out

    def training_step(self, batch, emb=None):
        images, labels = batch
        images, labels = images.to(self.device), labels.to(self.device)
        output = self(images, emb.to(self.device)) if self.embedding_size and emb is not None else self(images)
        loss = F.cross_entropy(output, labels)
        acc = accuracy(output, labels)
        return loss, acc

    def validation_step(self, batch, emb=None):
        images, labels = batch
        images, labels = images.to(self.device), labels.to(self.device)
        output = self(images, emb.to(self.device)) if self.embedding_size and emb is not None else self(images)
        loss = F.cross_entropy(output, labels)
        acc = accuracy(output, labels)
        return {"val_loss": loss, "val_acc": acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x["val_loss"] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x["val_acc"] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {"val_loss": epoch_loss.item(), "val_acc": epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print(f"Epoch [{epoch}], val_loss: {result['val_loss']:.4f}, val_acc: {result['val_acc']:.4f}")
        return result["val_loss"], result["val_acc"]




class HybridMnistModel(nn.Module):
    def __init__(self, device="cpu", embedding_size=0):
        """
        A hybrid model that first extracts CNN features from the image and then
        concatenates the quantum embedding.
        """
        super().__init__()
        # Define a simple CNN feature extractor.
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # input: [B,1,28,28] -> [B,16,28,28]
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> [B,16,14,14]
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # -> [B,32,14,14]
            nn.ReLU(),
            nn.MaxPool2d(2)   # -> [B,32,7,7]
        )
        cnn_output_size = 32 * 7 * 7
        total_input_size = cnn_output_size + embedding_size
        self.fc = nn.Linear(total_input_size, 10)
        self.device = device
        self.embedding_size = embedding_size

    def forward(self, image, emb=None):
        # Process image with CNN.
        features = self.cnn(image)  # Expecting image shape: [B,1,28,28]
        features = features.view(features.size(0), -1)
        if emb is not None:
            features = torch.cat((features, emb), dim=1)
        return self.fc(features)

    def training_step(self, batch, emb=None):
        images, labels = batch
        images, labels = images.to(self.device), labels.to(self.device)
        output = self(images, emb.to(self.device)) if self.embedding_size and emb is not None else self(images)
        loss = F.cross_entropy(output, labels)
        _, preds = torch.max(output, dim=1)
        acc = torch.tensor(torch.sum(preds == labels).item() / len(preds))
        return loss, acc

    def validation_step(self, batch, emb=None):
        images, labels = batch
        images, labels = images.to(self.device), labels.to(self.device)
        output = self(images, emb.to(self.device)) if self.embedding_size and emb is not None else self(images)
        loss = F.cross_entropy(output, labels)
        _, preds = torch.max(output, dim=1)
        acc = torch.tensor(torch.sum(preds == labels).item() / len(preds))
        return {"val_loss": loss, "val_acc": acc}

    def epoch_end(self, epoch, result):
        print(f"Epoch [{epoch}], val_loss: {result['val_loss']:.4f}, val_acc: {result['val_acc']:.4f}")
        return result["val_loss"], result["val_acc"]