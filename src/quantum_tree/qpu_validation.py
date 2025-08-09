import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import QNN
import argparse

from boson_sampler import BosonSampler
from utils import MNIST_partial
import pickle
import time

import perceval.providers.scaleway as scw
from torch.utils.data import Subset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")
_ = torch.manual_seed(1618)
torch.set_float32_matmul_precision('high')

parser = argparse.ArgumentParser(
    description="Evaluate a QNN model on a fraction of the MNIST validation set using a QPU session."
)
parser.add_argument("--scw_project_id", type=str,
                    help="SCW project ID.")
parser.add_argument("--scw_token", type=str,
                    help="SCW API Key.")
parser.add_argument("--filename", type=str, default="qnn.pkl",
                    help="Filename to load the pickled model and metrics from.")
parser.add_argument("--platform", type=str, default="qpu:ascella",
                    help="Platform identifier (e.g., 'qpu:ascella').")
parser.add_argument("--m", type=int, default=12,
                    help="Number of qumodes for the BosonSampler.")
parser.add_argument("--n", type=int, default=3,
                    help="Number of gate parameters for the BosonSampler.")
parser.add_argument("--fraction", type=float, default=0.2,
                    help="Fraction of the MNIST validation dataset to use for testing.")
parser.add_argument("--batch_size", type=int, default=2,
                    help="Test batch size.")
args = parser.parse_args()

__SCW_PROJECT_ID = args.scw_project_id
__SCW_TOKEN = args.scw_token
filename = args.filename
platform = args.platform
m = args.m
n = args.n
fraction = args.fraction
batch_size = args.batch_size

def get_session():
    return scw.Session(
        project_id=__SCW_PROJECT_ID,
        token=__SCW_TOKEN,
        platform=platform,
        max_idle_duration_s=12000,
        max_duration_s=36000,
    )

def initialize_qpu():
    global session, bs
    if session is not None:
        try:
            session.stop()
        except Exception:
            pass
    session = get_session()
    session.start()
    bs = BosonSampler(m, n, session=session)
    print("QPU session reinitialized.")

with open(filename, 'rb') as f:
    (train_loss_detailed, train_acc_detailed, train_acc, val_acc, 
        train_loss, val_loss, run_name, model_file, model_arch) = pickle.load(f)

session = None
bs = None

initialize_qpu()

model = QNN(bs, device).to(device)
model.load_state_dict(model_file)
print(model_arch)

val_dataset = MNIST_partial(split='val')
subset_size = int(len(val_dataset) * fraction)
subset_indices = list(range(subset_size))
subset_dataset = Subset(val_dataset, subset_indices)
test_loader = DataLoader(subset_dataset, batch_size=2, shuffle=False)

model.eval()
correct = 0
total = 0
loss_fn = nn.CrossEntropyLoss()
losses = []

batch_iterator = iter(test_loader)
batch_index = 0

while True:
    try:
        images, labels = next(batch_iterator)
    except StopIteration:
        break

    success = False
    attempts = 0
    while not success:
        try:
            images, labels = images.to(device), labels.to(device)
            outputs = model.evaluation(images)
            loss_val = loss_fn(outputs, labels).item()
            losses.append(loss_val)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            print(f"Batch {batch_index} - Loss: {loss_val:.6f}, Predicted: {predicted}, GT: {labels}")
            success = True
        except Exception as e:
            attempts += 1
            print(f"Error on batch {batch_index} (attempt {attempts}): {e}. Retrying after reinitializing QPU session...")
            time.sleep(10)
            initialize_qpu()
            model = QNN(bs, device).to(device)
            model.load_state_dict(model_file)
    batch_index += 1

accuracy = (correct / total) * 100
average_loss = sum(losses) / len(losses)
print(f"Test accuracy: {accuracy:.3f}%, Test loss: {average_loss:.6f}")

if session is not None:
    session.stop()